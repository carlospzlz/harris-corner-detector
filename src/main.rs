use eframe::egui;
use std::collections::HashSet;
use std::time::Duration;

const FRAME_SIZE: (usize, usize) = (1280, 720);

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let realsense_ctx =
        realsense_rust::context::Context::new().expect("Failed to create RealSense context");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([960.0, 550.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Harris Corner Detector",
        options,
        Box::new(|cc| Ok(Box::new(MyApp::new(cc, realsense_ctx)))),
    )
}

struct MyApp {
    pipeline: realsense_rust::pipeline::ActivePipeline,
}

impl MyApp {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        realsense_ctx: realsense_rust::context::Context,
    ) -> Self {
        let devices = realsense_ctx.query_devices(HashSet::new());
        let pipeline = realsense_rust::pipeline::InactivePipeline::try_from(&realsense_ctx)
            .expect("Failed to create inactive pipeline from context");
        let pipeline = start_pipeline(devices, pipeline);
        Self { pipeline }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, egui_ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Get frame
        let timeout = Duration::from_millis(100);
        let frames = match self.pipeline.wait(Some(timeout)) {
            Ok(frames) => Some(frames),
            Err(e) => {
                log::error!("{e}");
                None
            }
        };

        egui::CentralPanel::default().show(egui_ctx, |ui| {
            if let Some(frames) = frames {
                let mut ir_frames = frames.frames_of_type::<realsense_rust::frame::InfraredFrame>();
                let ir_frame = ir_frames.remove(0);
                let img = infrared_frame_to_gray_img(&ir_frame);
                let gradient_x = imageproc::gradients::horizontal_sobel(&img);
                let gradient_y = imageproc::gradients::vertical_sobel(&img);
                let gradient_img = combine_gradients_into_luma_img(&gradient_x, &gradient_y);
                let size = ui.available_size();
                let (width, height) = (size[0].round() as u32, size[1].round() as u32);
                let gradient_img = image::DynamicImage::ImageLuma8(gradient_img);
                let gradient_img = gradient_img
                    .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
                    .to_rgb8();
                let img = egui::ColorImage::from_rgb(
                    [width as usize, height as usize],
                    gradient_img.as_raw(),
                );
                let texture = egui_ctx.load_texture("sobel", img, Default::default());
                ui.image(&texture);
            }
        });

        egui_ctx.request_repaint();
    }
}

/// Starts RealSense pipeline
fn start_pipeline(
    devices: Vec<realsense_rust::device::Device>,
    pipeline: realsense_rust::pipeline::InactivePipeline,
) -> realsense_rust::pipeline::ActivePipeline {
    let realsense_device = find_realsense(devices);

    if realsense_device.is_none() {
        log::error!("No RealSense device found!");
        std::process::exit(-1);
    }

    let mut config = realsense_rust::config::Config::new();
    let realsense_device = realsense_device.unwrap();
    let serial_number = realsense_device
        .info(realsense_rust::kind::Rs2CameraInfo::SerialNumber)
        .unwrap();
    config
        .enable_device_from_serial(serial_number)
        .expect("Failed to enable device")
        .disable_all_streams()
        .expect("Failed to disable all streams")
        .enable_stream(
            realsense_rust::kind::Rs2StreamKind::Infrared,
            Some(1),
            FRAME_SIZE.0,
            FRAME_SIZE.1,
            realsense_rust::kind::Rs2Format::Y8,
            30,
        )
        .expect("Failed to enable infrared stream");

    let pipeline = pipeline
        .start(Some(config))
        .expect("Failed to start pipeline");

    for mut sensor in pipeline.profile().device().sensors() {
        // Disable emitter
        if sensor.supports_option(realsense_rust::kind::Rs2Option::EmitterEnabled) {
            sensor
                .set_option(realsense_rust::kind::Rs2Option::EmitterEnabled, 0.0)
                .expect("Failed to set option: EmitterEnabled");
        }
        // Enable Auto Exposure
        if sensor.supports_option(realsense_rust::kind::Rs2Option::EnableAutoExposure) {
            sensor
                .set_option(realsense_rust::kind::Rs2Option::EnableAutoExposure, 1.0)
                .expect("Failed to set option: EnableAutoExposure");
        }
    }

    pipeline
}

/// Finds first Real Sense device available
fn find_realsense(
    devices: Vec<realsense_rust::device::Device>,
) -> Option<realsense_rust::device::Device> {
    for device in devices {
        let name = match_info(&device, realsense_rust::kind::Rs2CameraInfo::Name);
        if name.starts_with("Intel RealSense") {
            return Some(device);
        }
    }
    None
}

/// Gets info from a device or returns "N/A"
fn match_info(
    device: &realsense_rust::device::Device,
    info_param: realsense_rust::kind::Rs2CameraInfo,
) -> String {
    match device.info(info_param) {
        Some(s) => String::from(s.to_str().unwrap()),
        None => String::from("N/A"),
    }
}

/// Convert InfraredFrame into GrayImage
fn infrared_frame_to_gray_img(frame: &realsense_rust::frame::InfraredFrame) -> image::GrayImage {
    let mut img = image::GrayImage::new(frame.width() as u32, frame.height() as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        match frame.get_unchecked(x as usize, y as usize) {
            realsense_rust::frame::PixelKind::Y8 { y } => {
                *pixel = image::Luma([*y]);
            }
            _ => panic!("Color type is wrong!"),
        }
    }
    img
}

/// Convert InfraredFrame into RgbImage
fn infrared_frame_to_rgb_img(frame: &realsense_rust::frame::InfraredFrame) -> image::RgbImage {
    let mut img = image::RgbImage::new(frame.width() as u32, frame.height() as u32);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        match frame.get_unchecked(x as usize, y as usize) {
            realsense_rust::frame::PixelKind::Y8 { y } => {
                *pixel = image::Rgb([*y, *y, *y]);
            }
            _ => panic!("Color type is wrong!"),
        }
    }
    img
}

/// Combine x and y gradients for visualisation
fn combine_gradients_into_luma_img(
    gradient_x: &image::ImageBuffer<image::Luma<i16>, Vec<i16>>,
    gradient_y: &image::ImageBuffer<image::Luma<i16>, Vec<i16>>,
) -> image::GrayImage {
    if (gradient_x.width() != gradient_y.width()) || (gradient_x.height() != gradient_y.height()) {
        panic!("Gradient images of different size!");
    }

    let (width, height) = (gradient_x.width(), gradient_x.height());

    let mut img = image::GrayImage::new(width, height);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        // Think about the gradient as a vector where gx and gy are the
        // components, so computing the magnitude will show how much of a
        // border a pixel is, not only horizontal/vertical but also diagonal
        // or any direction.
        let gx = gradient_x.get_pixel(x, y).0[0];
        let gy = gradient_y.get_pixel(x, y).0[0];
        let magnitude = ((gx as f32).powi(2) + (gy as f32).powi(2)).sqrt();
        let value = magnitude.clamp(0.0, 255.0);
        *pixel = image::Luma([value as u8]);
    }
    img
}
