use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Context};
use candle::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::depth_anything_v2::{DepthAnythingV2, DepthAnythingV2Config};
use candle_transformers::models::dinov2;
use enterpolation::Generator;

const DINO_IMG_SIZE: usize = 518;
const MAGIC_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const MAGIC_STD: [f32; 3] = [0.229, 0.224, 0.225];

static CONTEXT: OnceLock<Mutex<DepthBridge>> = OnceLock::new();
static EMPTY_C_STRING: &[u8; 1] = b"\0";

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

struct DepthBridge {
    device: Device,
    model: DepthModel,
}

struct DepthModel(DepthAnythingV2);

unsafe impl Send for DepthModel {}
unsafe impl Sync for DepthModel {}

struct RenderedImage {
    buffer: Vec<u8>,
    width: u32,
    height: u32,
}

struct AssetLayout {
    dinov2_weights: PathBuf,
    depth_anything_weights: PathBuf,
}

impl AssetLayout {
    fn new(root: PathBuf) -> Result<Self, CandleDepthStatusCode> {
        if !root.exists() {
            return Err(set_error(
                CandleDepthStatusCode::AssetRootMissing,
                format!("asset directory {:?} does not exist", root),
            ));
        }
        let dinov2_weights = root.join("dinov2_vits14.fp16.safetensors");
        if !dinov2_weights.exists() {
            return Err(set_error(
                CandleDepthStatusCode::AssetDinov2Missing,
                format!(
                    "missing dinov2_vits14.fp16.safetensors under {:?}",
                    dinov2_weights
                ),
            ));
        }
        let depth_anything_weights = root.join("depth_anything_v2_vits.fp16.safetensors");
        if !depth_anything_weights.exists() {
            return Err(set_error(
                CandleDepthStatusCode::AssetDepthModelMissing,
                format!(
                    "missing depth_anything_v2_vits.fp16.safetensors under {:?}",
                    depth_anything_weights
                ),
            ));
        }
        Ok(Self {
            dinov2_weights,
            depth_anything_weights,
        })
    }
}

#[repr(C)]
pub struct CandleDepthInitOptions {
    pub asset_dir: *const c_char,
}

#[repr(C)]
pub struct CandleDepthImageView {
    pub data: *const u8,
    pub len: usize,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

#[repr(C)]
pub struct CandleDepthRequest {
    pub image: CandleDepthImageView,
    pub use_color_map: u8,
}

#[repr(C)]
pub struct CandleDepthImage {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandleDepthStatusCode {
    Ok = 0,
    InvalidArgument = 1,
    NotInitialized = 2,
    AlreadyInitialized = 3,
    RuntimeError = 4,
    AssetRootMissing = 5,
    AssetDinov2Missing = 6,
    AssetDepthModelMissing = 7,
    LoadDinov2Failed = 8,
    LoadDepthModelFailed = 9,
    MetalUnavailable = 10,
}

#[no_mangle]
pub unsafe extern "C" fn candle_depth_init(
    options: *const CandleDepthInitOptions,
) -> CandleDepthStatusCode {
    if options.is_null() {
        return set_error(
            CandleDepthStatusCode::InvalidArgument,
            "options pointer was null",
        );
    }
    if CONTEXT.get().is_some() {
        return set_error(
            CandleDepthStatusCode::AlreadyInitialized,
            "bridge already initialised",
        );
    }
    let options = &*options;
    let asset_dir = match c_string_to_path(options.asset_dir) {
        Ok(path) => path,
        Err(code) => return code,
    };
    match initialise(&asset_dir) {
        Ok(context) => {
            let _ = CONTEXT.set(Mutex::new(context));
            CandleDepthStatusCode::Ok
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub extern "C" fn candle_depth_is_ready() -> bool {
    CONTEXT.get().is_some()
}

#[no_mangle]
pub unsafe extern "C" fn candle_depth_infer(
    request: *const CandleDepthRequest,
    out_image: *mut CandleDepthImage,
) -> CandleDepthStatusCode {
    if request.is_null() {
        return set_error(
            CandleDepthStatusCode::InvalidArgument,
            "request pointer was null",
        );
    }
    if out_image.is_null() {
        return set_error(
            CandleDepthStatusCode::InvalidArgument,
            "out_image pointer was null",
        );
    }
    let request = &*request;
    let context = match CONTEXT.get() {
        Some(ctx) => ctx,
        None => {
            return set_error(
                CandleDepthStatusCode::NotInitialized,
                "bridge not initialised",
            )
        }
    };
    let mut guard = context.lock().expect("mutex poisoned");
    match run_inference(&mut guard, request) {
        Ok(rendered) => {
            let mut output = rendered.buffer;
            let image = CandleDepthImage {
                data: output.as_mut_ptr(),
                len: output.len(),
                capacity: output.capacity(),
                width: rendered.width,
                height: rendered.height,
                channels: 4,
            };
            std::mem::forget(output);
            *out_image = image;
            CandleDepthStatusCode::Ok
        }
        Err(err) => set_error(CandleDepthStatusCode::RuntimeError, err),
    }
}

#[no_mangle]
pub unsafe extern "C" fn candle_depth_free_image(image: *mut CandleDepthImage) {
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
pub unsafe extern "C" fn candle_depth_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        if let Some(message) = slot.borrow().as_ref() {
            message.as_ptr()
        } else {
            EMPTY_C_STRING.as_ptr() as *const c_char
        }
    })
}

#[cfg(feature = "metal")]
fn initialise(asset_dir: &PathBuf) -> Result<DepthBridge, CandleDepthStatusCode> {
    let layout = AssetLayout::new(asset_dir.to_path_buf())?;

    let device = Device::new_metal(0).map_err(|e| {
        set_error(
            CandleDepthStatusCode::MetalUnavailable,
            format!("failed to acquire Metal device: {e}"),
        )
    })?;

    let dinov2_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[layout.dinov2_weights.clone()], DType::F16, &device)
    }
    .map_err(|err| {
        set_error(
            CandleDepthStatusCode::LoadDinov2Failed,
            format!("failed to load DINOv2 weights: {err}"),
        )
    })?;
    let dinov2 = dinov2::vit_small(dinov2_vb).map_err(|err| {
        set_error(
            CandleDepthStatusCode::LoadDinov2Failed,
            format!("failed to build DINOv2 model: {err:?}"),
        )
    })?;

    let depth_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[layout.depth_anything_weights.clone()],
            DType::F16,
            &device,
        )
    }
    .map_err(|err| {
        set_error(
            CandleDepthStatusCode::LoadDepthModelFailed,
            format!("failed to load Depth Anything weights: {err}"),
        )
    })?;
    let depth_anything = DepthAnythingV2::new(
        Arc::new(dinov2),
        DepthAnythingV2Config::vit_small(),
        depth_vb,
    )
    .map_err(|err| {
        set_error(
            CandleDepthStatusCode::LoadDepthModelFailed,
            format!("failed to build Depth Anything model: {err:?}"),
        )
    })?;

    Ok(DepthBridge {
        device,
        model: DepthModel(depth_anything),
    })
}

#[cfg(not(feature = "metal"))]
fn initialise(asset_dir: &PathBuf) -> Result<DepthBridge, CandleDepthStatusCode> {
    let _ = asset_dir;
    Err(set_error(
        CandleDepthStatusCode::MetalUnavailable,
        "bridge compiled without Metal support",
    ))
}

fn run_inference(
    context: &mut DepthBridge,
    request: &CandleDepthRequest,
) -> anyhow::Result<RenderedImage> {
    let input = prepare_input(&request.image, &context.device)?;

    let depth = context.model.0.forward(&input)?;
    // Move to CPU for post-processing and work in f32 for the CPU pipeline.
    let depth = depth.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let colormap = SpectralRColormap::new();
    let output = post_process_image(&depth, request.use_color_map != 0, &colormap)?;
    let (_, height, width) = output
        .dims3()
        .context("depth output should have shape (3, h, w)")?;
    let buffer = tensor_to_rgba(&output)?;
    Ok(RenderedImage {
        buffer,
        width: width as u32,
        height: height as u32,
    })
}

fn prepare_input(view: &CandleDepthImageView, device: &Device) -> anyhow::Result<Tensor> {
    if view.data.is_null() {
        return Err(anyhow!("input image data pointer was null"));
    }
    let width = view.width as usize;
    let height = view.height as usize;
    let channels = view.channels as usize;
    if channels != 3 && channels != 4 {
        return Err(anyhow!("expected 3 or 4 channels, got {channels}"));
    }
    let element_count = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(channels))
        .ok_or_else(|| anyhow!("image dimensions overflow"))?;
    if view.len != element_count {
        return Err(anyhow!(
            "input image length {} did not match expected {}",
            view.len,
            element_count
        ));
    }
    let data = unsafe { std::slice::from_raw_parts(view.data, view.len).to_vec() };
    let dyn_img = if channels == 4 {
        let buffer = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_vec(
            width as u32,
            height as u32,
            data,
        )
        .ok_or_else(|| anyhow!("failed to build RGBA image buffer"))?;
        image::DynamicImage::ImageRgba8(buffer)
    } else {
        let buffer = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_vec(
            width as u32,
            height as u32,
            data,
        )
        .ok_or_else(|| anyhow!("failed to build RGB image buffer"))?;
        image::DynamicImage::ImageRgb8(buffer)
    };

    let resized = dyn_img.resize_to_fill(
        DINO_IMG_SIZE as u32,
        DINO_IMG_SIZE as u32,
        image::imageops::FilterType::Triangle,
    );
    let rgb = resized.to_rgb8();
    let data = rgb.into_raw();

    let tensor = Tensor::from_vec(data, (DINO_IMG_SIZE, DINO_IMG_SIZE, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?;

    let max_pixel_val = Tensor::try_from(255.0f32)?
        .to_device(&Device::Cpu)?
        .broadcast_as(tensor.shape())?;
    let tensor = (tensor / max_pixel_val)?;
    let tensor = normalize_image(&tensor, &MAGIC_MEAN, &MAGIC_STD)?;
    let tensor = tensor.to_dtype(DType::F16)?;
    Ok(tensor.to_device(device)?)
}

fn normalize_image(image: &Tensor, mean: &[f32; 3], std: &[f32; 3]) -> candle::Result<Tensor> {
    let mean_tensor =
        Tensor::from_vec(mean.to_vec(), (3, 1, 1), &image.device())?.broadcast_as(image.shape())?;
    let std_tensor =
        Tensor::from_vec(std.to_vec(), (3, 1, 1), &image.device())?.broadcast_as(image.shape())?;
    image.sub(&mean_tensor)?.div(&std_tensor)
}

fn post_process_image(
    depth: &Tensor,
    use_color_map: bool,
    colormap: &SpectralRColormap,
) -> anyhow::Result<Tensor> {
    let out = scale_image(depth)?;

    let out = if use_color_map {
        colormap.gray_to_color(&out)?
    } else {
        let slices = [&out, &out, &out];
        Tensor::cat(&slices, 0)?
    };

    let max_pixel_val = Tensor::try_from(255.0f32)?
        .to_device(out.device())?
        .broadcast_as(out.shape())?;
    let out = (out * max_pixel_val)?;

    Ok(out.to_dtype(DType::U8)?)
}

fn scale_image(depth: &Tensor) -> anyhow::Result<Tensor> {
    let depth = depth.squeeze(0)?;
    let flat_values: Vec<f32> = depth.flatten_all()?.to_vec1()?;

    let (min_val, max_val) = flat_values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, &value| {
            (acc.0.min(value), acc.1.max(value))
        });

    let min_val_tensor = Tensor::try_from(min_val)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;
    let depth = (depth - min_val_tensor)?;

    let range = (max_val - min_val).max(f32::EPSILON);
    let range_tensor = Tensor::try_from(range)?
        .to_device(depth.device())?
        .broadcast_as(depth.shape())?;

    Ok((depth / range_tensor)?)
}

fn tensor_to_rgba(image: &Tensor) -> anyhow::Result<Vec<u8>> {
    let image = image.to_device(&Device::Cpu)?;
    let (channels, _, _) = image
        .dims3()
        .context("expected output image with 3 channels")?;
    if channels != 3 {
        return Err(anyhow!(
            "expected 3 channels in output image, got {channels}"
        ));
    }
    let image = image.permute((1, 2, 0))?.flatten_all()?;
    let rgb = image.to_vec1::<u8>()?;
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.extend_from_slice(chunk);
        rgba.push(255);
    }
    Ok(rgba)
}

fn c_string_to_path(ptr: *const c_char) -> Result<PathBuf, CandleDepthStatusCode> {
    c_string_to_string(ptr).map(PathBuf::from)
}

fn c_string_to_string(ptr: *const c_char) -> Result<String, CandleDepthStatusCode> {
    if ptr.is_null() {
        return Err(set_error(
            CandleDepthStatusCode::InvalidArgument,
            "received null string pointer",
        ));
    }
    let c_str = unsafe { CStr::from_ptr(ptr) };
    let str_slice = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => {
            return Err(set_error(
                CandleDepthStatusCode::InvalidArgument,
                "string pointer was not valid UTF-8",
            ))
        }
    };
    Ok(str_slice.to_string())
}

fn set_error(
    code: CandleDepthStatusCode,
    message: impl std::fmt::Display,
) -> CandleDepthStatusCode {
    let message = message.to_string();
    LAST_ERROR.with(|slot| {
        let c_string =
            CString::new(message).unwrap_or_else(|_| CString::new("failed to set error").unwrap());
        *slot.borrow_mut() = Some(c_string);
    });
    code
}

struct SpectralRColormap {
    gradient: enterpolation::linear::ConstEquidistantLinear<f32, palette::LinSrgb, 9>,
}

impl SpectralRColormap {
    fn new() -> Self {
        use palette::LinSrgb;
        // Same anchors as the example pipeline.
        let gradient =
            enterpolation::linear::ConstEquidistantLinear::<f32, _, 9>::equidistant_unchecked([
                LinSrgb::new(0.3686, 0.3098, 0.6353),
                LinSrgb::new(0.1961, 0.5333, 0.7412),
                LinSrgb::new(0.4000, 0.7608, 0.6471),
                LinSrgb::new(0.6706, 0.8667, 0.6431),
                LinSrgb::new(0.9020, 0.9608, 0.5961),
                LinSrgb::new(0.9961, 0.8784, 0.5451),
                LinSrgb::new(0.9922, 0.6824, 0.3804),
                LinSrgb::new(0.9569, 0.4275, 0.2627),
                LinSrgb::new(0.8353, 0.2431, 0.3098),
            ]);
        Self { gradient }
    }

    fn gray_to_color(&self, gray: &Tensor) -> anyhow::Result<Tensor> {
        let gray = gray.squeeze(0)?;
        let values: Vec<f32> = gray.flatten_all()?.to_vec1()?;
        let mut rgb_values = Vec::with_capacity(values.len() * 3);
        for v in values {
            let color = self.gradient.gen(v);
            rgb_values.push(color.red);
            rgb_values.push(color.green);
            rgb_values.push(color.blue);
        }
        let (_, height, width) = gray.dims3().context("expected gray image dims")?;
        let tensor = Tensor::from_vec(rgb_values, (height, width, 3), &Device::Cpu)?;
        Ok(tensor.permute((2, 0, 1))?)
    }
}
