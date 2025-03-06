use opencv::core::{Mat, MatTraitConst, Rect};
use crate::detectors::mean_squared_error::MeanSquaredError;

pub struct MseSubdivide {
    dimensions: (usize, usize),                 // subdivision count: ( row,  column )
    resolution: (usize, usize),                 // frame resolution: eg. ( 1280, 720 )
    cells: Vec<MeanSquaredError>,               // MSE objects
    region_dims: (
        Vec<Rect>,                            // pre-computed regions for cropping frame
        (usize, usize),                         // frame resolution (width, height)
        usize                                   // count of how many regions there are
    )
}

impl MseSubdivide {
    pub fn new(dimensions: (usize, usize), resolution: (usize, usize), frame: Mat) -> Self {
        let region_dims = Self::calculate_regions(dimensions, resolution);
        let crop_region = Mat::roi(&frame, Rect::new(
            0,0, region_dims.1.0 as i32, region_dims.1.1 as i32
        )).unwrap().try_clone().unwrap();

        // init cells
        let mut cells = Vec::with_capacity(region_dims.2);
        for region in region_dims.0.iter() {
            cells.push(MeanSquaredError::new(crop_region.clone()));
        }

        Self {
            dimensions,
            resolution,
            cells,
            region_dims,
        }
    }

    pub fn calculate_region_dimension(
        dimensions: (usize, usize),
        resolution: (usize, usize)
    ) -> (usize, usize) {
        (resolution.0 / dimensions.0, resolution.1 / dimensions.1)
    }

    pub fn calculate_regions(
        dimensions: (usize, usize),
        resolution: (usize, usize)
    ) -> (
        Vec<Rect>,
        (usize, usize),
        usize
    ) {
        let region_dims = Self::calculate_region_dimension(dimensions, resolution);
        let region_count = dimensions.0 * dimensions.1;
        let mut regions: Vec<Rect> = Vec::with_capacity(region_count);

        let mut start_x;
        let mut start_y;
        let mut end_x;
        let mut end_y;
        let bounding_dims = (dimensions.0 - 1, dimensions.1 - 1);

        for y in 0..dimensions.1 {
            for x in 0..dimensions.0 {
                start_x = x * region_dims.0;
                start_y = y * region_dims.1;

                // if a resolution is not divisible by 2 without remainders,
                // expand the row or col so that it fits the rest of the area
                end_x = if x == bounding_dims.0 { resolution.0 } else { start_x + region_dims.0 };
                end_y = if y == bounding_dims.1 { resolution.1 } else { start_y + region_dims.1 };

                // TODO: Review this code, it was funky

                regions.push(Rect::new(
                    start_x as i32,
                    start_y as i32,
                    (end_x - start_x) as i32,
                    (end_y - start_y) as i32
                ));
            }

        }
        (regions,
         region_dims,
         region_count)
    }
    
    fn crop_region(&mut self, frame: Mat, region: Rect) -> Mat {
        Mat::roi(&frame, region).unwrap().try_clone().unwrap()
    }

    // supplying a value for threshold will compose a frame from all cells except those which do not
    // have values exceeding the threshold.
    pub fn get_frame(&self, threshold: Option<f64>) -> Mat {
        let mut frame = Mat::new_rows_cols_with_default(
            self.resolution.1 as i32,
            self.resolution.0 as i32,
            opencv::core::CV_8UC1,                  // Assuming a single-channel 8-bit image
            opencv::core::Scalar::all(0.0),
        ).unwrap();

        let mut cell_frame;
        let mut roi;

        if let Some(threshold) = threshold {
            for (cell, region) in self.cells.iter().zip(self.region_dims.0.iter()) {
                if cell.value >= threshold {
                    cell_frame = cell.diff_mask.clone();
                } else {
                    cell_frame = Mat::default();
                }
                roi = Mat::roi_mut(&mut frame, *region).unwrap();
                cell_frame.copy_to(&mut roi).unwrap();
            }
        } else {
            for (cell, region) in self.cells.iter().zip(self.region_dims.0.iter()) {
                cell_frame = cell.diff_mask.clone();
                roi = Mat::roi_mut(&mut frame, *region).unwrap();
                cell_frame.copy_to(&mut roi).unwrap();
            }
        }

        return frame;
    }

    pub fn get_region_frame(&self, index: usize) -> Option<Mat> {
        if index >= self.cells.len() {
            return None;
        }
        let cell = &self.cells[index];
        Some(cell.diff_mask.clone())
    }

    pub fn calculate_mse(&mut self, cur_frame: &opencv::core::Mat) {
        let mut cropped_frame;
        for i in 0..self.cells.len() {
            cropped_frame = self.crop_region(
                cur_frame.clone(),
                self.region_dims.0[i]
            );
            self.cells[i].update(cropped_frame);
        }
    }
    
    pub fn update(&mut self, cur_frame: Mat) {
        self.calculate_mse(&cur_frame);
    }

}