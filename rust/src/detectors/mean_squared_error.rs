use opencv::core::{
    self,
    Mat,
    MatExprTraitConst,
    MatTraitConst
};
use opencv::Error;
use crate::util::video_frames::{FrameProcessor, VideoFrames};

const SAMPLE_COUNT: usize = 10;

pub struct MeanSquaredError {
    diff_mask: Mat,
    sqr_diff: Mat,

    num_pixels: f64,
    multiply_scale: f64,
    dtype: i32,
    mse_values: Vec<f64>,
}

impl<'a> MeanSquaredError {
    pub fn new(prv_frame: &'a Mat) -> Self {
        Self {
            diff_mask: Mat::zeros(
                prv_frame.rows(),
                prv_frame.cols(),
                prv_frame.typ(),
            ).unwrap().to_mat().unwrap(),
            sqr_diff: Mat::zeros(
                prv_frame.rows(),
                prv_frame.cols(),
                prv_frame.typ(),
            ).unwrap().to_mat().unwrap(),
            num_pixels: (prv_frame.rows() * prv_frame.cols()) as f64,
            multiply_scale: 1.0,
            dtype: -1,
            mse_values: vec![0.0; SAMPLE_COUNT],
        }
    }

    pub fn default() -> Self {
        Self {
            diff_mask: Mat::default(),
            sqr_diff: Mat::default(),
            num_pixels: 0.0,
            multiply_scale: 1.0,
            dtype: -1,
            mse_values: Vec::default(),
        }
    }

    pub fn get_diff_mask(&self) -> &Mat {
        &self.diff_mask
    }

    pub fn get_value(&self) -> f64 {
        self.mse_values[SAMPLE_COUNT - 1]
    }

    pub fn get_value_avg(&self) -> f64 {
        self.mse_values.iter().sum::<f64>() / self.mse_values.len() as f64
    }

    fn _calculate_mse(&mut self, prv_frame: &'a Mat, cur_frame: &'a Mat) -> Result<(), Error> {

        core::absdiff(
            &prv_frame,
            cur_frame,
            &mut self.diff_mask
        )?;

        self.sqr_diff = self.diff_mask.clone();
        core::multiply(
            &self.diff_mask,
            &self.diff_mask,
            &mut self.sqr_diff,
            self.multiply_scale,
            self.dtype
        )?;

        self.mse_values.remove(0);
        self.mse_values.push(core::sum_elems(&self.diff_mask)?[0] / self.num_pixels);

        Ok(())
    }
}

impl<'a> FrameProcessor<'a> for MeanSquaredError {
    fn update(&mut self, video_frames: &VideoFrames) -> Result<(), Error> {
        self._calculate_mse(&video_frames.mono.quarter.prev, &video_frames.mono.quarter.cur)
    }
}