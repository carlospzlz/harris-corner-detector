use eframe::egui;
use std::collections::HashSet;
use std::time::Duration;

//const FRAME_SIZE: (usize, usize) = (1280, 720);
const FRAME_SIZE: (usize, usize) = (848, 480);

#[derive(Debug, PartialEq)]
enum DebugImage {
    Sobel,
    R,
    NMS,
}

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let realsense_ctx =
        realsense_rust::context::Context::new().expect("Failed to create RealSense context");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 800.0]),
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
    window_size: u32,
    k: f32,
    nms_window_size: u32,
    radius: u32,
    debug_img: DebugImage,
    r_colormap_threshold1: f32,
    r_colormap_threshold2: f32,
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
        Self {
            pipeline,
            window_size: 7,
            k: 0.05,
            nms_window_size: 20,
            radius: 3,
            debug_img: DebugImage::R,
            r_colormap_threshold1: 0.33,
            r_colormap_threshold2: 0.66,
        }
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

        egui::SidePanel::left("left_panel")
            //.exact_width(130.0)
            .show(egui_ctx, |ui| {
                ui.label("Harris window size:");
                ui.add(egui::Slider::new(&mut self.window_size, 3..=20));
                ui.label("K:");
                ui.add(egui::Slider::new(&mut self.k, 0.04..=0.06));
                ui.label("NMS window size:");
                ui.add(egui::Slider::new(&mut self.nms_window_size, 3..=100));
                ui.label("Visualization radius:");
                ui.add(egui::Slider::new(&mut self.radius, 3..=20));
                ui.horizontal(|ui| {
                    ui.label("Debug");
                    let separator = egui::Separator::default();
                    ui.add(separator.horizontal());
                });

                egui::ComboBox::from_label("")
                    .selected_text(format!("{:?}", self.debug_img))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.debug_img, DebugImage::Sobel, "Sobel");
                        ui.selectable_value(&mut self.debug_img, DebugImage::R, "R");
                        ui.selectable_value(&mut self.debug_img, DebugImage::NMS, "NMS");
                    });

                ui.label("R colormap");
                ui.add(
                    egui::Slider::new(&mut self.r_colormap_threshold1, 0.0..=0.33)
                );
                ui.add(
                    egui::Slider::new(&mut self.r_colormap_threshold2, 0.0..=0.33)
                );
            });

        egui::CentralPanel::default().show(egui_ctx, |ui| {
            if let Some(frames) = frames {
                let mut ir_frames = frames.frames_of_type::<realsense_rust::frame::InfraredFrame>();
                let ir_frame = ir_frames.remove(0);
                let ir_img = infrared_frame_to_gray_img(&ir_frame);
                let gradient_x = imageproc::gradients::horizontal_sobel(&ir_img);
                let gradient_y = imageproc::gradients::vertical_sobel(&ir_img);
                let gradient_img = combine_gradients_into_luma_img(&gradient_x, &gradient_y);
                let response_img =
                    compute_corner_response(gradient_x, gradient_y, self.window_size, self.k);
                let corners = non_maximal_suppression(&response_img, self.nms_window_size);

                let asize = ui.available_size();
                let size = ((asize[0].round()) as u32, (asize[1].round() / 2.0) as u32);
                ui.vertical(|ui| {
                    // IR image with corners
                    let img = identify_corners(ir_img.clone(), &corners, self.radius);
                    add_image_frame_item(egui_ctx, ui, "IR image".to_string(), img, size);

                    match self.debug_img {
                        DebugImage::Sobel => add_image_frame_item(
                            egui_ctx,
                            ui,
                            "Sobel".to_string(),
                            gradient_img,
                            size,
                        ),
                        DebugImage::R => {
                            let img = apply_colormap(
                                response_img,
                                self.r_colormap_threshold1,
                                self.r_colormap_threshold2,
                            );
                            add_image_frame_item(egui_ctx, ui, "R".to_string(), img, size)
                        }
                        DebugImage::NMS => {
                            let img = create_nms_img(&corners, ir_img.width(), ir_img.height());
                            add_image_frame_item(egui_ctx, ui, "NMS".to_string(), img, size)
                        }
                    }
                });
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

/// Combine x and y gradients for visualisation
fn combine_gradients_into_luma_img(
    gradient_x: &image::ImageBuffer<image::Luma<i16>, Vec<i16>>,
    gradient_y: &image::ImageBuffer<image::Luma<i16>, Vec<i16>>,
) -> image::RgbImage {
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
    image::DynamicImage::ImageLuma8(img).to_rgb8()
}

/// This is the heart of this methodology
fn compute_corner_response(
    gradient_x: image::ImageBuffer<image::Luma<i16>, Vec<i16>>,
    gradient_y: image::ImageBuffer<image::Luma<i16>, Vec<i16>>,
    window_size: u32,
    k: f32,
) -> image::GrayImage {
    if (gradient_x.width() != gradient_y.width()) || (gradient_x.height() != gradient_y.height()) {
        panic!("Gradient images of different size!");
    }

    let (width, height) = (gradient_x.width(), gradient_x.height());
    let half_window_size = window_size / 2.0 as u32;

    let mut corner_response = vec![0.0; (width * height) as usize];
    let mut max_value = 0.0;
    for x in half_window_size..(width - 1) - half_window_size {
        for y in half_window_size..(height - 1) - half_window_size {
            let mut gx_square_sum = 0;
            let mut gy_square_sum = 0;
            let mut cross_product_sum = 0;
            for xx in x - half_window_size..x + half_window_size {
                for yy in y - half_window_size..y + half_window_size {
                    let gx = gradient_x.get_pixel(xx, yy).0[0];
                    let gy = gradient_y.get_pixel(xx, yy).0[0];
                    gx_square_sum += (gx as i32).pow(2);
                    gy_square_sum += (gy as i32).pow(2);
                    cross_product_sum += (gx as i32) * (gy as i32);
                }
            }

            // Formulas from Shree Nayar video, professor in Columbia University
            // https://www.youtube.com/watch?v=Z_HwkG90Yvw&t=440s
            let a = gx_square_sum as f32;
            let b = 2.0 * cross_product_sum as f32;
            let c = gy_square_sum as f32;

            let trace = a + c;
            let sqrt_term = ((a - c).powi(2) + 4.0 * b.powi(2)).sqrt();
            let lambda1 = (trace + sqrt_term) / 2.0;
            let lambda2 = (trace - sqrt_term) / 2.0;

            let r = lambda1 * lambda2 - k * (lambda1 + lambda2).powi(2);
            corner_response[(x * height + y) as usize] = r;

            max_value = r.max(max_value);
            //println!("{max_value}");
        }
    }

    let mut img = image::GrayImage::new(width, height);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let normalized = corner_response[(x * height + y) as usize] / max_value;
        *pixel = image::Luma([(normalized * 255.0) as u8]);
    }

    img
}

fn apply_colormap(r: image::GrayImage, threshold1: f32, threshold2: f32) -> image::RgbImage {
    let mut img = image::RgbImage::new(r.width(), r.height());
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let normalized = r.get_pixel(x, y).0[0] as f32 / 255.0;
        *pixel = colormap(normalized, threshold1, threshold2);
    }
    img
}

/// Implement color map
/// Black -> Red -> Yellow -> White
fn colormap(value: f32, threshold1: f32, threshold2: f32) -> image::Rgb<u8> {
    let v = value.clamp(0.0, 1.0);

    let (r, g, b) = if v < threshold1 {
        lerp_color(v, 0.0, (0, 0, 0), threshold1, (255, 0, 0)) // Black → Red
    } else if v < threshold2 {
        lerp_color(v, threshold1, (255, 0, 0), threshold2, (255, 255, 0)) // Red → Yellow
    } else {
        lerp_color(v, threshold2, (255, 255, 0), 1.0, (255, 255, 255)) // Yellow → Black
    };

    image::Rgb([r, g, b])
}

/// Linearly interpolates between two colors based on value position.
fn lerp_color(
    value: f32,
    v_min: f32,
    c_min: (u8, u8, u8),
    v_max: f32,
    c_max: (u8, u8, u8),
) -> (u8, u8, u8) {
    let t = ((value - v_min) / (v_max - v_min)).clamp(0.0, 1.0);
    (
        (c_min.0 as f32 + t * (c_max.0 as f32 - c_min.0 as f32)) as u8,
        (c_min.1 as f32 + t * (c_max.1 as f32 - c_min.1 as f32)) as u8,
        (c_min.2 as f32 + t * (c_max.2 as f32 - c_min.2 as f32)) as u8,
    )
}

fn non_maximal_suppression(r: &image::GrayImage, window_size: u32) -> Vec<(u32, u32)> {
    let mut corners = Vec::<(u32, u32)>::new();

    let (width, height) = (r.width(), r.height());
    let half_window_size = window_size / 2.0 as u32;

    for x in half_window_size..(width - 1) - half_window_size {
        for y in half_window_size..(height - 1) - half_window_size {
            let central_value = r.get_pixel(x, y).0[0];
            let is_peak = || -> bool {
                if central_value == 0 {
                    return false;
                }
                for xx in x - half_window_size..x + half_window_size {
                    for yy in y - half_window_size..y + half_window_size {
                        let value = r.get_pixel(xx, yy).0[0];
                        if xx == x && yy == y {
                            continue;
                        }
                        if value > central_value {
                            return false;
                        }
                    }
                }
                return true;
            }();
            if is_peak {
                corners.push((x, y));
            }
        }
    }

    corners
}

fn create_nms_img(corners: &Vec<(u32, u32)>, widht: u32, height: u32) -> image::RgbImage {
    let mut img = image::RgbImage::new(widht, height);
    for (x, y) in corners {
        img.put_pixel(*x, *y, image::Rgb([255, 255, 255]));
    }
    img
}

fn identify_corners(
    ir_img: image::GrayImage,
    corners: &Vec<(u32, u32)>,
    radius: u32,
) -> image::RgbImage {
    let mut img = image::DynamicImage::ImageLuma8(ir_img).to_rgb8();
    for (x, y) in corners {
        img = imageproc::drawing::draw_hollow_circle(
            &img,
            (*x as i32, *y as i32),
            radius as i32,
            image::Rgb([0, 255, 0]),
        );
        img = imageproc::drawing::draw_hollow_circle(
            &img,
            (*x as i32, *y as i32),
            (radius + 1) as i32,
            image::Rgb([0, 255, 0]),
        );
    }

    img
}

fn add_image_frame_item(
    egui_ctx: &egui::Context,
    ui: &mut egui::Ui,
    title: String,
    img: image::RgbImage,
    size: (u32, u32),
) {
    // Account for title
    let size = (size.0, size.1 - 30);
    let img = image::DynamicImage::ImageRgb8(img);
    let img = img
        .resize_exact(size.0, size.1, image::imageops::FilterType::Lanczos3)
        .to_rgb8();
    let img = egui::ColorImage::from_rgb([size.0 as usize, size.1 as usize], img.as_raw());
    egui::Frame::canvas(ui.style()).show(ui, |ui| {
        ui.vertical(|ui| {
            ui.label(title.clone());
            let texture = egui_ctx.load_texture(title, img, Default::default());
            ui.image(&texture);
        });
    });
}
