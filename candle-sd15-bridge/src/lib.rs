use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context};
use candle::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::clip;
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use tokenizers::Tokenizer;

const HEIGHT: usize = 512;  //512;
const WIDTH: usize = 512;   //512;
const GUIDANCE_SCALE: f64 = 7.5;
const NEGATIVE_PROMPT_DEFAULT: &str = "";

static CONTEXT: OnceLock<Mutex<Sd15Bridge>> = OnceLock::new();
static EMPTY_C_STRING: &[u8; 1] = b"\0";

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

struct Sd15Bridge {
    device: Device,
    unet_dtype: DType,
    vae_dtype: DType,
    config: stable_diffusion::StableDiffusionConfig,
    layout: AssetLayout,
}

struct AssetLayout {
    tokenizer: PathBuf,
    clip: PathBuf,
    unet: PathBuf,
    vae: PathBuf,
}

impl AssetLayout {
    fn new(root: PathBuf) -> Result<Self, CandleSdStatusCode> {
        if !root.exists() {
            return Err(set_error(
                CandleSdStatusCode::AssetRootMissing,
                format!("asset directory {:?} does not exist", root),
            ));
        }
        let layout = Self {
            tokenizer: root.join("tokenizer/tokenizer.json"),
            clip: root.join("text_encoder/model.fp16.safetensors"),
            unet: root.join("unet/diffusion_pytorch_model.fp16.safetensors"),
            vae: root.join("vae/diffusion_pytorch_model.fp16.safetensors"),
        };
        if !layout.tokenizer.exists() {
            return Err(set_error(
                CandleSdStatusCode::AssetTokenizerMissing,
                format!(
                    "missing asset tokenizer/tokenizer.json at {:?}",
                    layout.tokenizer
                ),
            ));
        }
        if !layout.clip.exists() {
            return Err(set_error(
                CandleSdStatusCode::AssetClipMissing,
                format!(
                    "missing asset text_encoder/model.fp16.safetensors at {:?}",
                    layout.clip
                ),
            ));
        }
        if !layout.unet.exists() {
            return Err(set_error(
                CandleSdStatusCode::AssetUnetMissing,
                format!(
                    "missing asset unet/diffusion_pytorch_model.fp16.safetensors at {:?}",
                    layout.unet
                ),
            ));
        }
        if !layout.vae.exists() {
            return Err(set_error(
                CandleSdStatusCode::AssetVaeMissing,
                format!(
                    "missing asset vae/diffusion_pytorch_model.fp16.safetensors at {:?}",
                    layout.vae
                ),
            ));
        }
        Ok(layout)
    }
}

#[repr(C)]
pub struct CandleSdInitOptions {
    pub asset_dir: *const c_char,
    pub use_metal: u8,
}

#[repr(C)]
pub struct CandleSdRequest {
    pub prompt: *const c_char,
    pub negative_prompt: *const c_char,
    pub steps: u32,
    pub seed: u64,
    pub use_seed: u8,
}

#[repr(C)]
pub struct CandleSdImage {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandleSdStatusCode {
    Ok = 0,
    InvalidArgument = 1,
    NotInitialized = 2,
    AlreadyInitialized = 3,
    RuntimeError = 4,
    AssetDirInvalid = 5,
    AssetRootMissing = 6,
    AssetTokenizerMissing = 7,
    AssetClipMissing = 8,
    AssetUnetMissing = 9,
    AssetVaeMissing = 10,
    LoadTokenizerFailed = 11,
    LoadClipFailed = 12,
    LoadUnetFailed = 13,
    LoadVaeFailed = 14,
    MetalUnavailable = 15,
}

#[no_mangle]
pub unsafe extern "C" fn candle_sd_init(
    options: *const CandleSdInitOptions,
) -> CandleSdStatusCode {
    if options.is_null() {
        return set_error(
            CandleSdStatusCode::InvalidArgument,
            "options pointer was null",
        );
    }
    if CONTEXT.get().is_some() {
        return set_error(
            CandleSdStatusCode::AlreadyInitialized,
            "bridge already initialised",
        );
    }
    let options = &*options;
    let asset_dir = match c_string_to_path(options.asset_dir) {
        Ok(path) => path,
        Err(code) => return code,
    };
    let use_metal = options.use_metal != 0;

    match initialise(&asset_dir, use_metal) {
        Ok(context) => {
            let _ = CONTEXT.set(Mutex::new(context));
            CandleSdStatusCode::Ok
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub extern "C" fn candle_sd_is_ready() -> bool {
    CONTEXT.get().is_some()
}

#[no_mangle]
pub unsafe extern "C" fn candle_sd_generate(
    request: *const CandleSdRequest,
    out_image: *mut CandleSdImage,
) -> CandleSdStatusCode {
    log_process_memory("candle_sd_generate");
    if request.is_null() {
        return set_error(
            CandleSdStatusCode::InvalidArgument,
            "request pointer was null",
        );
    }
    if out_image.is_null() {
        return set_error(
            CandleSdStatusCode::InvalidArgument,
            "out_image pointer was null",
        );
    }
    let context = match CONTEXT.get() {
        Some(ctx) => ctx,
        None => {
            return set_error(
                CandleSdStatusCode::NotInitialized,
                "bridge not initialised",
            )
        }
    };
    let request = &*request;
    let prompt = match c_string_to_string(request.prompt) {
        Ok(p) => p,
        Err(code) => return code,
    };
    if prompt.trim().is_empty() {
        return set_error(
            CandleSdStatusCode::InvalidArgument,
            "prompt must not be empty",
        );
    }
    let negative_prompt = if request.negative_prompt.is_null() {
        NEGATIVE_PROMPT_DEFAULT.to_string()
    } else {
        match c_string_to_string(request.negative_prompt) {
            Ok(s) => s,
            Err(code) => return code,
        }
    };
    let steps = request.steps.max(1) as usize;
    let seed = if request.use_seed != 0 {
        Some(request.seed)
    } else {
        None
    };

    log_process_memory(format!("args parsed, seed {seed:?}").as_str());

    let mut guard = context.lock().expect("mutex poisoned");
    match generate_image(&mut guard, &prompt, &negative_prompt, steps, seed) {
        Ok(buffer) => {
            let mut buffer = buffer;
            let image = CandleSdImage {
                data: buffer.as_mut_ptr(),
                len: buffer.len(),
                capacity: buffer.capacity(),
                width: WIDTH as u32,
                height: HEIGHT as u32,
                channels: 4,
            };
            std::mem::forget(buffer);
            *out_image = image;
            CandleSdStatusCode::Ok
        }
        Err(err) => set_error(CandleSdStatusCode::RuntimeError, err),
    }
}

#[no_mangle]
pub unsafe extern "C" fn candle_sd_free_image(image: *mut CandleSdImage) {
    if image.is_null() {
        return;
    }
    let image = &mut *image;
    if image.data.is_null() {
        return;
    }
    let data = image.data;
    let len = image.len;
    let capacity = image.capacity;
    let _ = Vec::from_raw_parts(data, len, capacity);
    image.data = std::ptr::null_mut();
    image.len = 0;
    image.capacity = 0;
}

#[no_mangle]
pub unsafe extern "C" fn candle_sd_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        if let Some(message) = slot.borrow().as_ref() {
            message.as_ptr()
        } else {
            EMPTY_C_STRING.as_ptr() as *const c_char
        }
    })
}

fn initialise(asset_dir: &Path, use_metal: bool) -> Result<Sd15Bridge, CandleSdStatusCode> {
    let layout = AssetLayout::new(asset_dir.to_path_buf())?;
    let device = if use_metal {
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(d) => d,
                Err(e) => return Err(set_error(CandleSdStatusCode::MetalUnavailable, e)),
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            return Err(set_error(
                CandleSdStatusCode::MetalUnavailable,
                "bridge compiled without Metal support",
            ));
        }
    } else {
        Device::Cpu
    };

    let (unet_dtype, vae_dtype) = if matches!(device, Device::Cpu) {
        (DType::F32, DType::F32)
    } else {
        // (DType::F16, DType::F32)
        (DType::F16, DType::F32)
    };
    let config = stable_diffusion::StableDiffusionConfig::v1_5(None, Some(HEIGHT), Some(WIDTH));

    Ok(Sd15Bridge {
        device,
        unet_dtype,
        vae_dtype,
        config,
        layout,
    })
}

fn generate_image(
    context: &mut Sd15Bridge,
    prompt: &str,
    negative_prompt: &str,
    steps: usize,
    seed: Option<u64>,
) -> anyhow::Result<Vec<u8>> {
    log_process_memory("generate_image");
    let mut scheduler = context.config.build_scheduler(steps)?;
    let text_embeddings = build_text_embeddings(context, prompt, negative_prompt)?;
    let use_guidance = GUIDANCE_SCALE > 1.0;
    log_process_memory("embeddings computed");

    let latent_shape = (1, 4, context.config.height / 8, context.config.width / 8);
    let mut latents = sample_latents(latent_shape, &context.device, context.unet_dtype, seed)?;
    latents = (latents * scheduler.init_noise_sigma())?.to_dtype(context.unet_dtype)?;
    log_process_memory("latents initialised");

    log_process_memory(match &context.device {
        Device::Cpu => "device cpu",
        Device::Metal(_) => "device metal",
        Device::Cuda(_) => "device cuda",
    });
    let unet = match context.load_unet() {
        Ok(u) => u,
        Err(err) => {
            log_process_memory(format!("unet load failed: {err:?}").as_str());
            return Err(err);
        }
    };
    log_process_memory(format!("unet loaded dtype={:?}", context.unet_dtype).as_str());

    let timesteps = scheduler.timesteps().to_vec();
    for (step_idx, &timestep) in timesteps.iter().enumerate() {
        log_process_memory(format!("unet step {}/{steps} begin", step_idx + 1).as_str());
        let mut latent_model_input = latents.clone();
        if use_guidance {
            latent_model_input = Tensor::cat(&[&latent_model_input, &latent_model_input], 0)?;
        }
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
        log_process_memory("latent input scaled");

        // context.device.synchronize().ok();
        let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;
        // context.device.synchronize().ok();
        let noise_pred = if use_guidance {
            let chunks = noise_pred.chunk(2, 0)?;
            let noise_pred_uncond = &chunks[0];
            let noise_pred_text = &chunks[1];
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * GUIDANCE_SCALE)?)?
        } else {
            noise_pred
        };

        log_process_memory("noise_pred computed");

        latents = scheduler
            .step(&noise_pred, timestep, &latents)?
            .to_dtype(context.unet_dtype)?;
        log_process_memory(format!("unet step {}/{steps} END", step_idx + 1).as_str());
        drop(noise_pred);
    }
    log_process_memory("unet completed");
    drop(unet);
    drop(scheduler);
    drop(text_embeddings);

    context.device.synchronize().ok();
    let latents = (latents / 0.18215)?.to_dtype(context.vae_dtype)?;
    log_process_memory("vae will load");
    let vae = context.load_vae()?;
    log_process_memory("vae did load");
    let image = vae.decode(&latents)?;
    drop(vae);
    let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    log_process_memory("vae decoded");
    let image = (image.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    let image = image.i(0)?;
    let image = image.permute((1, 2, 0))?.flatten_all()?;
    let rgb = image.to_vec1::<u8>()?;
    Ok(to_rgba(&rgb))
}

fn build_text_embeddings(
    context: &Sd15Bridge,
    prompt: &str,
    negative_prompt: &str,
) -> anyhow::Result<Tensor> {
    let tokenizer = context.load_tokenizer()?;
    let clip = context.load_clip()?;

    let prompt_tokens = tokenize_prompt(&tokenizer, &context.config.clip, prompt, &context.device)?;
    let prompt_embedding = clip.forward(&prompt_tokens)?;

    let negative_tokens = tokenize_prompt(
        &tokenizer,
        &context.config.clip,
        if negative_prompt.is_empty() {
            NEGATIVE_PROMPT_DEFAULT
        } else {
            negative_prompt
        },
        &context.device,
    )?;
    let negative_embedding = clip.forward(&negative_tokens)?;

    let text_embeddings = Tensor::cat(&[negative_embedding, prompt_embedding], 0)?;
    Ok(text_embeddings.to_dtype(context.unet_dtype)?)
}

fn tokenize_prompt(
    tokenizer: &Tokenizer,
    config: &clip::Config,
    prompt: &str,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)
        .context("failed to encode prompt")?
        .get_ids()
        .to_vec();
    let pad_id = match &config.pad_with {
        Some(pad) => tokenizer
            .get_vocab(true)
            .get(pad.as_str())
            .copied()
            .ok_or_else(|| anyhow!("pad token {} not found in tokenizer", pad))?,
        None => tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .copied()
            .ok_or_else(|| anyhow!("<|endoftext|> not found in tokenizer"))?,
    };
    if tokens.len() > config.max_position_embeddings {
        return Err(anyhow!(
            "prompt is too long: {} tokens > max {}",
            tokens.len(),
            config.max_position_embeddings
        ));
    }
    while tokens.len() < config.max_position_embeddings {
        tokens.push(pad_id)
    }
    Ok(Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?)
}

fn to_rgba(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.extend_from_slice(chunk);
        rgba.push(255);
    }
    rgba
}

fn sample_latents(
    shape: (usize, usize, usize, usize),
    device: &Device,
    dtype: DType,
    seed: Option<u64>,
) -> anyhow::Result<Tensor> {
    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = StandardNormal;
        let elem = shape.0 * shape.1 * shape.2 * shape.3;
        let mut data = vec![0f32; elem];
        for value in &mut data {
            *value = normal.sample(&mut rng);
        }
        let tensor = Tensor::from_vec(data, shape, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok(tensor)
    } else {
        let tensor = Tensor::randn(0f32, 1f32, shape, device)?;
        Ok(tensor.to_dtype(dtype)?)
    }
}

fn log_process_memory(stage: &str) {
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    unsafe {
        use libc::{mach_msg_type_number_t, mach_task_basic_info};
        use libc::{task_info, KERN_SUCCESS, MACH_TASK_BASIC_INFO, MACH_TASK_BASIC_INFO_COUNT};

        let mut info = std::mem::zeroed::<mach_task_basic_info>();
        let mut count: mach_msg_type_number_t = MACH_TASK_BASIC_INFO_COUNT;

        #[allow(deprecated)]
        let task = libc::mach_task_self();

        let result = task_info(
            task,
            MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut i32,
            &mut count,
        );

        if result == KERN_SUCCESS {
            println!(
                "[sd15][{stage}] resident={} MB, resident_max={}, virtual={} MB",
                info.resident_size / 1024 / 1024,
                info.resident_size_max / 1024 / 1024,
                info.virtual_size / 1024 / 1024
            );
        }
    }

    #[cfg(not(any(target_os = "ios", target_os = "macos")))]
    {
        println!("[sd15][{stage}] (process stats not available on this platform)");
    }
}

impl Sd15Bridge {
    fn load_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        // self.device.wait_until_completed().ok();
        // self.device.synchronize().ok();
        Tokenizer::from_file(&self.layout.tokenizer)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("failed to load tokenizer at {:?}", self.layout.tokenizer))
    }

    fn load_clip(&self) -> anyhow::Result<clip::ClipTextTransformer> {
        // self.device.synchronize().ok();
        stable_diffusion::build_clip_transformer(
            &self.config.clip,
            &self.layout.clip,
            &self.device,
            DType::F32,
        )
        .with_context(|| format!("failed to load clip weights at {:?}", self.layout.clip))
    }

    fn load_unet(&self) -> anyhow::Result<UNet2DConditionModel> {
        // self.device.synchronize().ok();
        self.config
            .build_unet(&self.layout.unet, &self.device, 4, false, self.unet_dtype)
            .with_context(|| format!("failed to load unet at {:?}", self.layout.unet))
    }

    fn load_vae(&self) -> anyhow::Result<AutoEncoderKL> {
        // self.device.synchronize().ok();
        self.config
            .build_vae(&self.layout.vae, &self.device, self.vae_dtype)
            .with_context(|| format!("failed to load vae at {:?}", self.layout.vae))
    }
}

fn c_string_to_path(ptr: *const c_char) -> Result<PathBuf, CandleSdStatusCode> {
    c_string_to_string(ptr).map(PathBuf::from)
}

fn c_string_to_string(ptr: *const c_char) -> Result<String, CandleSdStatusCode> {
    if ptr.is_null() {
        return Err(set_error(
            CandleSdStatusCode::InvalidArgument,
            "received null string pointer",
        ));
    }
    let c_str = unsafe { CStr::from_ptr(ptr) };
    match c_str.to_str() {
        Ok(s) => Ok(s.to_string()),
        Err(_) => Err(set_error(
            CandleSdStatusCode::InvalidArgument,
            "string was not valid UTF-8",
        )),
    }
}

fn set_error(code: CandleSdStatusCode, message: impl std::fmt::Display) -> CandleSdStatusCode {
    let msg = message.to_string();
    let cstring =
        CString::new(msg).unwrap_or_else(|_| CString::new("error message contained nul").unwrap());
    LAST_ERROR.with(|slot| {
        slot.replace(Some(cstring));
    });
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_appends_alpha() {
        let rgb = vec![0u8, 1, 2, 10, 11, 12];
        let rgba = to_rgba(&rgb);
        assert_eq!(rgba, vec![0, 1, 2, 255, 10, 11, 12, 255]);
    }
}
