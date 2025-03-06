use opencv::core::{Mat, MatTraitConst, Scalar, Size};
use opencv::{imgproc, videoio, Error};
use opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT;
use opencv::prelude::VideoCaptureTrait;

pub trait FrameProcessor <'a> {
    fn update(
        &mut self,
        video_frames: &VideoFrames,
    ) -> Result<(), Error>;
}

pub struct Frame {
    pub cur: Mat,
    pub prev: Mat,
}

impl Frame {
    pub fn new(size: Size, mat_type: i32) -> Self {
        Self {
            cur: Mat::new_size_with_default(
                size,
                mat_type,
                Scalar::default()
            ).unwrap(),
            prev: Mat::new_size_with_default(
                size,
                mat_type,
                Scalar::default()
            ).unwrap(),
        }
    }
    pub fn invalidate(&mut self) {
        std::mem::swap(&mut self.prev, &mut self.cur);
    }
    pub fn reset(&mut self) -> Result<(), Error> {
        self.cur = Mat::new_size_with_default(
            self.cur.size()?,
            self.cur.typ(),
            Scalar::default()
        )?;
        self.prev = Mat::new_size_with_default(
            self.prev.size()?,
            self.prev.typ(),
            Scalar::default()
        )?;
        Ok(())
    }
}

pub struct FrameSamples {
    pub full: Frame,
    pub half: Frame,
    pub quarter: Frame,
}

impl FrameSamples {
    pub fn new(
        size_full: Size,
        size_half: Size,
        size_quarter: Size,
        mat_type: i32,
    ) -> Self {
        Self {
            full: Frame::new(size_full, mat_type),
            half: Frame::new(size_half, mat_type),
            quarter: Frame::new(size_quarter, mat_type),
        }
    }
    pub fn invalidate(&mut self) {
        self.full.invalidate();
        self.half.invalidate();
        self.quarter.invalidate();
    }
    pub fn reset(&mut self) -> Result<(), Error> {
        self.full.reset()?;
        self.half.reset()?;
        self.quarter.reset()?;
        Ok(())
    }
}

pub struct VideoFrames {
    pub color: FrameSamples,
    pub mono: FrameSamples,

    size_full: Size,
    size_half: Size,
    size_quarter: Size,
}

impl VideoFrames {
    pub fn new(
        size_full: Size,
        size_half: Size,
        size_quarter: Size,
    ) -> Self {
        Self {
            color: FrameSamples::new(
                size_full,
                size_half,
                size_quarter,
                opencv::core::CV_8UC3,
            ),
            mono: FrameSamples::new(
                size_full,
                size_half,
                size_quarter,
                opencv::core::CV_8UC1,
            ),
            size_full,
            size_half,
            size_quarter,
        }
    }

    pub fn invalidate(&mut self) {
        self.color.invalidate();
        self.mono.invalidate();
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        self.color.reset()?;
        self.mono.reset()?;
        Ok(())
    }

    pub fn read_frame(&mut self, cam: &mut videoio::VideoCapture) -> opencv::Result<(), Error> {

        cam.read(&mut self.color.full.cur)?;

        if self.color.full.cur.empty() {
            return Err(Error::new(
                opencv::core::StsError,
                String::from("No frames left in video."),
            ));
        }

        // invalidate all frames
        self.invalidate();

        // COLOR \\
        imgproc::resize(
            &self.color.full.cur,
            &mut self.color.half.cur,
            self.size_half,
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        imgproc::resize(
            &self.color.half.cur,
            &mut self.color.quarter.cur,
            self.size_quarter,
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        // MONOCHROME \\
        imgproc::cvt_color(
            &self.color.full.cur,
            &mut self.mono.full.cur,
            imgproc::COLOR_BGR2GRAY,
            0,
            ALGO_HINT_DEFAULT
        )?;

        imgproc::cvt_color(
            &self.color.half.cur,
            &mut self.mono.half.cur,
            imgproc::COLOR_BGR2GRAY,
            0,
            ALGO_HINT_DEFAULT
        )?;

        imgproc::cvt_color(
            &self.color.quarter.cur,
            &mut self.mono.quarter.cur,
            imgproc::COLOR_BGR2GRAY,
            0,
            ALGO_HINT_DEFAULT
        )?;

        // BLUR \\
        let mut blur_mono_half = self.mono.half.cur.clone();
        imgproc::gaussian_blur(
            &self.mono.half.cur,
            &mut blur_mono_half,
            Size::new(5, 5),
            0.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
            ALGO_HINT_DEFAULT,
        )?;
        self.mono.half.cur = blur_mono_half;

        let mut blur_mono_quarter = self.mono.quarter.cur.clone();
        imgproc::gaussian_blur(
            &self.mono.quarter.cur,
            &mut blur_mono_quarter,
            Size::new(5, 5),
            0.0,
            0.0,
            opencv::core::BORDER_DEFAULT,
            ALGO_HINT_DEFAULT,
        )?;
        self.mono.quarter.cur = blur_mono_quarter;

        Ok(())
    }
}

