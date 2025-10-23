use std::fmt::Display;

use comfy_table::presets::UTF8_NO_BORDERS;
use comfy_table::{Cell, ContentArrangement, Table};
use modron::{ComputableValue, Die};
use plotters::chart::ChartBuilder;
use plotters::coord::Shift;
use plotters::evcxr::{evcxr_figure, SVGWrapper};
use plotters::prelude::{DrawingArea, DrawingAreaErrorKind, DrawingBackend, IntoSegmentedCoord};
use plotters::series::Histogram;
use plotters::style::{Color, RGBColor, WHITE};

#[allow(type_alias_bounds)]
pub type HistResult<DB: DrawingBackend> = Result<(), DrawingAreaErrorKind<DB::ErrorType>>;

const HIST_COLOR: RGBColor = RGBColor(0xEF, 0x97, 0x06);
const MAX_X_LABELS: usize = 45;

pub trait PrintExt {
    fn table(&self) -> String;

    fn plot<DB>(&self, title: &str, area: DrawingArea<DB, Shift>) -> HistResult<DB>
    where
        DB: DrawingBackend;

    fn print_table(&self) {
        print!("{}", self.table());
    }

    fn evcxr_plot(&self, title: &str) -> SVGWrapper {
        evcxr_figure((640, 480), |area| {
            self.plot(title, area)?;
            Ok(())
        })
    }
}

impl<T> PrintExt for Die<T>
where
    T: ComputableValue + Display,
{
    fn table(&self) -> String {
        let pb = self.probabilities();
        let mean = self.mean();
        let stddev = self.stddev();
        let denom = self.denom();

        let mut table = Table::new();
        table
            .load_preset(UTF8_NO_BORDERS)
            .set_content_arrangement(ContentArrangement::DynamicFullWidth)
            .set_header(vec![Cell::new("Значение"), Cell::new("Вероятность")]);
        for (value, prob) in self.values().iter().zip(pb) {
            table.add_row(vec![
                Cell::new(value.to_string()),
                Cell::new(format!("{:6.2}%", prob * 100.0)),
            ]);
        }

        format!("Среднее: {mean:.3}±{stddev:.3} | Исходы: {denom}\n\n{table}\n")
    }

    fn plot<DB>(&self, title: &str, area: DrawingArea<DB, Shift>) -> HistResult<DB>
    where
        DB: DrawingBackend,
    {
        let cv = self.computed_values();
        let pb = self.probabilities();
        let mean = self.mean();
        let stddev = self.stddev();
        let x0 = *cv.iter().min().unwrap();
        let x1 = *cv.iter().max().unwrap();
        let y1 = pb
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .copied()
            .unwrap_or(1.0);

        area.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&area);
        chart
            .x_label_area_size(35)
            .y_label_area_size(50)
            .margin(10)
            .caption(
                format!("{title} | Среднее: {mean:.3}±{stddev:.3}"),
                ("sans-serif", 16.0),
            );

        let mut coord = chart.build_cartesian_2d((x0..x1).into_segmented(), 0f64..y1)?;

        coord
            .configure_mesh()
            .disable_x_mesh()
            .x_labels(cv.len().min(MAX_X_LABELS))
            .y_label_formatter(&|y: &f64| format!("{:5.1}%", *y * 100.0))
            .x_desc("Значение")
            .y_desc("Вероятность")
            .draw()?;

        coord.draw_series(
            Histogram::vertical(&coord)
                .style(HIST_COLOR.filled())
                .margin(1)
                .data(cv.into_iter().zip(pb)),
        )?;

        Ok(())
    }
}
