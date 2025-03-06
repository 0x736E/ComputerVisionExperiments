use std::fs;
use std::path::Path;
use std::process::exit;
use opencv::core::Size;
use opencv::highgui::imshow;
use opencv::{highgui, videoio};
use opencv::hub_prelude::{VideoCaptureTrait, VideoCaptureTraitConst};
use opencv::videoio::VideoCapture;
use structopt::StructOpt;

use crate::detectors::mean_squared_error::MeanSquaredError;
use crate::detectors::motion_mog2::MotionMog2;
use crate::masks::motion_overlay::MotionOverlay;
use crate::masks::overlay::OverlayProcessor;
use crate::util::stop_watch::StopWatch;
use crate::util::video_frames::{FrameProcessor, VideoFrames};


#[derive(StructOpt, Debug)]
#[structopt(name = "RustyVision", about = "OpenCV-based motion detection")]
pub struct VideoConfig {

    #[structopt(long, parse(from_os_str))]
    pub input: std::path::PathBuf,

    #[structopt(long)]
    pub silent: bool,

    #[structopt(long)]
    pub verbose: bool,

    #[structopt(long)]
    pub headless: bool,

    #[structopt(long, default_value = "2.0")]
    pub target_fps: f64,

    #[structopt(long, default_value = "0.4")]
    pub mse_threshold: f64,

    #[structopt(long, default_value = "5000.0")]
    pub mog2_threshold: f64,

    #[structopt(long, default_value = "500")]
    pub mog2_history: i32,

    #[structopt(long, default_value = "50.0")]
    pub mog2_sensitivity: f64,

    #[structopt(long, default_value = "17")]
    pub adaptive_block_size: i32,

    #[structopt(long, default_value = "9")]
    pub adaptive_c: f64,
}

pub struct VideoProcessor {
    cam: VideoCapture,
    video_frames: VideoFrames,
    mse_detector: MeanSquaredError,
    mog2_detector: MotionMog2,
    stopwatch: StopWatch,
    video_fps: f64,
    frame_skip: i32,
    frame_counter: i32,
    read_frame_retry_count: i32,
    motion_detected: bool,
}

impl VideoProcessor {

    pub fn new() -> Self {
        Self {
            cam: VideoCapture::default().unwrap(),
            video_frames: VideoFrames::new(
                Size::new(3840, 2160),
                Size::new(1280, 720),
                Size::new(640, 360),
            ),
            mse_detector: MeanSquaredError::default(),
            mog2_detector: MotionMog2::default(),
            stopwatch: StopWatch::new(),
            video_fps: 0.0,
            frame_skip: 0,
            frame_counter: 0,
            read_frame_retry_count: 0,
            motion_detected: false,
        }
    }

    pub fn load_videos(&mut self, file_path: &Path, conf: &VideoConfig) -> opencv::Result<()> {
        if file_path.is_dir() {
            for entry in fs::read_dir(&file_path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();

                if path.is_dir() {
                    continue
                }

                if let Some(extension) = path.extension() {
                    if  extension == "mp4" ||
                        extension == "avi" ||
                        extension == "mov" ||
                        extension == "mkv" ||
                        extension == "webm"
                    {
                        if !conf.silent {
                            println!("Processing file: {:?}", path);
                        }
                        if let Err(e) = self.process_video(path.to_str().unwrap(), conf) {
                            eprintln!("Failed to process file {:?}: {}", path, e);
                        }
                    }
                }
            }
            Ok(())
        } else {
            self.process_video(file_path.to_str().unwrap(), conf)
        }
    }

    pub fn process_video(&mut self, file_path: &str, conf: &VideoConfig) -> opencv::Result<()> {

        self.cam = videoio::VideoCapture::from_file(
            file_path,
            videoio::CAP_ANY
        )?;

        // Check if video file opened successfully
        if !self.cam.is_opened()? {
            panic!("Unable to open video file: {}", file_path);
        }

        // initialize
        self.video_frames = VideoFrames::new(
            Size::new(
                self.cam.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32,
                self.cam.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32
            ),
            Size::new(1280, 720),
            Size::new(640, 360),
        );
        self.video_fps = self.cam.get(videoio::CAP_PROP_FPS)?;
        self.frame_skip = (self.video_fps / conf.target_fps).ceil() as i32;
        self.frame_counter = 0;

        // initialize by reading the first frame
        loop {
            if self.video_frames.read_frame(&mut self.cam).is_ok() {
                self.mse_detector = MeanSquaredError::new(&self.video_frames.mono.quarter.cur);
                self.mog2_detector = MotionMog2::new(
                    &self.video_frames.mono.quarter.cur,
                    conf.mog2_history,
                    conf.mog2_sensitivity,
                    conf.adaptive_block_size,
                    conf.adaptive_c,
                );
                break;
            }
            self.read_frame_retry_count += 1;
            if self.read_frame_retry_count > 9 {
                panic!("Unable to read first frame from video file. Exiting!");
            }
        }

        // read the rest of the frames
        loop {

            self.frame_counter += 1;
            if self.frame_counter % self.frame_skip != 0 {
                self.cam.grab()?;
                continue;
            }

            if !conf.silent && conf.verbose {
                println!(
                    "Frame: {} of {}",
                    self.frame_counter,
                    self.cam.get(videoio::CAP_PROP_FRAME_COUNT)?
                );
            }

            self.motion_detected = false;
            self.stopwatch.start();

            if self.video_frames.read_frame(&mut self.cam).is_ok() {
                self.stopwatch.lap("Read Frame");

                self.mse_detector.update(&self.video_frames)?;
                self.stopwatch.lap("MSE");

                let mse_avg = self.mse_detector.get_value_avg();
                if mse_avg >= conf.mse_threshold {
                    self.mog2_detector.update(&self.video_frames)?;
                    self.stopwatch.lap("MOG2");
                    let mog2_avg = self.mog2_detector.get_area_avg();

                    if mog2_avg >= conf.mog2_threshold {
                        if !conf.silent && conf.verbose {
                            println!(
                                "Motion detected (Frame {}, MSE: {:.4}, MOG2:{:.0})",
                                self.frame_counter,
                                mse_avg,
                                mog2_avg
                            );
                        }
                    self.motion_detected = true;
                    }
                }

            } else {
                self.stopwatch.stop();
                if !conf.silent {
                    if conf.verbose {
                        println!("{}", self.stopwatch.to_string_detailed());
                    } else {
                        println!("{}", self.stopwatch.to_string());
                    }
                }
                return Ok(());
            }

            if !conf.headless {
                if self.motion_detected {
                    let mut motion_overlay = MotionOverlay::new(&self.mog2_detector);
                    let overlay_frame = motion_overlay.draw(&self.video_frames.color.half.cur)?;
                    self.stopwatch.lap("Overlay");
                    imshow("video", &overlay_frame)?;
                } else {
                    imshow("video", &self.video_frames.color.half.cur)?;
                }
                // imshow("mask", mog2_detector.get_diff_mask())?;

                match highgui::wait_key(1)? {

                    // ESC => exit
                    27 => {
                        self.stopwatch.stop();
                        if !conf.silent {
                            if conf.verbose {
                                println!("{}", self.stopwatch.to_string_detailed());
                            } else {
                                println!("{}", self.stopwatch.to_string());
                            }
                        }
                        exit(0);
                    }

                    // q => skip video
                    113 => {
                        self.stopwatch.stop();
                        if !conf.silent {
                            if conf.verbose {
                                println!("{}", self.stopwatch.to_string_detailed());
                            } else {
                                println!("{}", self.stopwatch.to_string());
                            }
                        }
                        return Ok(());
                    }

                    _ => {}
                }
            }

            self.stopwatch.tick();
        }
    }
}