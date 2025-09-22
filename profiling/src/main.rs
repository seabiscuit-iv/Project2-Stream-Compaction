use std::{fs::{File, OpenOptions}, io::{self, BufRead}, sync::{Arc, RwLock}, thread};

use eframe::*;
use egui::*;
use egui_plot::*;
use regex::Regex;
use screenshots::Screen;
use tokio::{
    io::AsyncBufReadExt
};
use std::io::Write;

const BLOCK_SIZES: [u32; 4] = [128, 256, 512, 1024];
const RESET_ALL: bool = true;

fn main() -> Result<(), eframe::Error> {
    let options = NativeOptions {  
        viewport: ViewportBuilder::default().with_fullscreen(true),
        ..Default::default()
    };

    for i in BLOCK_SIZES {
        if RESET_ALL {
            File::create(format!("profile_output/output_{}.txt", i)).unwrap();
        }
        if let Err(x) = 
            eframe::run_native(&format!("Project 2: Stream Compaction Profiler [Block Size: {i}]"), options.clone(), 
        Box::new(|_| Ok(Box::new(App::new(i))))
            )    
        {
            return Err(x);
        }
    }

    Ok(())
}

pub struct App {
    current: Tab,
    cpu_stream_compact_with_scan: Arc<RwLock<Vec<(f32, u32)>>>,
    cpu_stream_compact_without_scan: Arc<RwLock<Vec<(f32, u32)>>>,
    gpu_scan_naive: Arc<RwLock<Vec<(f32, u32)>>>,
    gpu_scan_efficient: Arc<RwLock<Vec<(f32, u32)>>>,
    gpu_stream_compact_efficient: Arc<RwLock<Vec<(f32, u32)>>>,
    gpu_scan_thrust: Arc<RwLock<Vec<(f32, u32)>>>,
    cpu_scan: Arc<RwLock<Vec<(f32, u32)>>>,
    block_size: u32
}

impl App {
    pub fn new(block_size: u32) -> Self {
        let plots = get_plots(block_size);

        Self { current: Tab::Scan, 
            cpu_stream_compact_with_scan: plots.0, 
            cpu_stream_compact_without_scan: plots.1,
            gpu_scan_naive: plots.2, 
            gpu_scan_efficient: plots.3,
            gpu_stream_compact_efficient: plots.4,
            gpu_scan_thrust: plots.5,
            cpu_scan: plots.6,
            block_size
        }
    }
}

fn get_plots(block_size: u32) -> (Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>, Arc<RwLock<Vec<(f32, u32)>>>) {
    let cpu_stream_compact_with_scan = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let cpu_stream_compact_without_scan = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let gpu_scan_naive: Arc<RwLock<Vec<(f32, u32)>>> = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let gpu_scan_efficient = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let gpu_stream_compact_efficient = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let gpu_scan_thrust = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));
    let cpu_scan = Arc::new(RwLock::new(Vec::<(f32, u32)>::new()));

    let cpu_stream_compact_with_scan_clone = cpu_stream_compact_with_scan.clone();
    let cpu_stream_compact_without_scan_clone = cpu_stream_compact_without_scan.clone();
    let gpu_scan_naive_clone = gpu_scan_naive.clone();
    let gpu_scan_efficient_clone = gpu_scan_efficient.clone();
    let gpu_stream_compact_efficient_clone = gpu_stream_compact_efficient.clone();
    let gpu_scan_thrust_clone = gpu_scan_thrust.clone();
    let cpu_scan_clone = cpu_scan.clone();


    thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();

        let file = File::open(format!("profile_output/output_{}.txt", block_size)).expect("Unable to open output file");
        let reader = io::BufReader::new(file);

        let mut prefetch_data: Vec<(f32, f32, f32, f32, f32, f32, f32, u32)> = Vec::new();

        for line in reader.lines() {
            let line = line.unwrap();
            let mut elems = line.split_whitespace();
            let cpu_sc_ms: f32 = elems.next().unwrap().parse().unwrap();
            let cpu_sc_wout_ms: f32 = elems.next().unwrap().parse().unwrap();
            let gpu_scan_ms: f32 = elems.next().unwrap().parse().unwrap();
            let gpu_scan_efficient_ms: f32 = elems.next().unwrap().parse().unwrap();
            let gpu_sc_efficient_ms: f32 = elems.next().unwrap().parse().unwrap();
            let gpu_scan_thrust_ms: f32 = elems.next().unwrap().parse().unwrap();
            let cpu_scan_ms: f32 = elems.next().unwrap().parse().unwrap();
            let size: u32 = elems.next().unwrap().parse().unwrap();

            prefetch_data.push((cpu_sc_ms, cpu_sc_wout_ms, gpu_scan_ms, gpu_scan_efficient_ms, gpu_sc_efficient_ms, gpu_scan_thrust_ms, cpu_scan_ms, size));
        }

        for i in 5..=27 {
            let size = 1 << i;

            let prefetch = prefetch_data.iter().find(|x| x.7 == i);

            if let Some(x) = prefetch {
                cpu_stream_compact_with_scan_clone.write().unwrap().push((x.0, x.7));
                cpu_stream_compact_without_scan_clone.write().unwrap().push((x.1, x.7));
                gpu_scan_naive_clone.write().unwrap().push((x.2, x.7));
                gpu_scan_efficient_clone.write().unwrap().push((x.3, x.7));
                gpu_stream_compact_efficient_clone.write().unwrap().push((x.4, x.7));
                gpu_scan_thrust_clone.write().unwrap().push((x.5, x.7));
                cpu_scan_clone.write().unwrap().push((x.6, x.7));

                continue;
            }

            let cpu_stream_compact_with_scan_move = cpu_stream_compact_with_scan_clone.clone();
            let cpu_stream_compact_without_scan_move = cpu_stream_compact_without_scan_clone.clone();
            let gpu_scan_naive_move = gpu_scan_naive_clone.clone();
            let gpu_scan_efficient_move = gpu_scan_efficient_clone.clone();
            let gpu_stream_compact_efficient_move = gpu_stream_compact_efficient_clone.clone();
            let gpu_scan_thrust_move = gpu_scan_thrust_clone.clone();
            let cpu_scan_move = cpu_scan_clone.clone();

            rt.block_on(async move {
                let mut file = OpenOptions::new();
                let mut file = file.write(true)   // open for writing
                    .append(true)  // append to the end instead of overwriting
                    .create(true)  // create if it doesn’t exist
                    .open(format!("profile_output/output_{}.txt", block_size)).expect("Unable to open output file");

                let value = run_tests(size, block_size).await;
                cpu_stream_compact_with_scan_move.write().unwrap().push((value.0, i));
                cpu_stream_compact_without_scan_move.write().unwrap().push((value.1, i));
                gpu_scan_naive_move.write().unwrap().push((value.2, i));
                gpu_scan_efficient_move.write().unwrap().push((value.3, i));
                gpu_stream_compact_efficient_move.write().unwrap().push((value.4, i));
                gpu_scan_thrust_move.write().unwrap().push((value.5, i));
                cpu_scan_move.write().unwrap().push((value.6, i));
                writeln!(file, "{} {} {} {} {} {} {} {}", value.0, value.1, value.2, value.3, value.4, value.5, value.6, i).unwrap();
            });
        }
    });

    (cpu_stream_compact_with_scan, cpu_stream_compact_without_scan, gpu_scan_naive, gpu_scan_efficient, gpu_stream_compact_efficient, gpu_scan_thrust, cpu_scan)
}




#[derive(PartialEq)]
enum Tab {
    Scan,
    StreamCompaction,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("tab_bar").show(ctx, |ui| {
            ui.add_space(2.0); // top padding
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.current, Tab::Scan, "Scan");
                ui.selectable_value(&mut self.current, Tab::StreamCompaction, "StreamCompaction");

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Screenshot").clicked() {
                        let screen = Screen::from_point(0, 0).unwrap();
                        let image = screen.capture().unwrap();
                        let tab = match self.current {
                            Tab::Scan => "scan",
                            Tab::StreamCompaction => "stream_compaction"
                        };
                        let file_name = format!("../img/{}_{}_old.png", tab, self.block_size);
                        image.save(file_name).unwrap();
                    }

                    if ui.button("Reset").clicked() {
                        File::create(format!("profile_output/output_{}.txt", self.block_size)).unwrap();
                    }
                });
            });
            ui.add_space(2.0); // top padding
        });

        
        let cpu_stream_compact_with_scan = self.cpu_stream_compact_with_scan.read().unwrap();
        let cpu_stream_compact_without_scan = self.cpu_stream_compact_without_scan.read().unwrap();
        let gpu_scan_naive = self.gpu_scan_naive.read().unwrap();
        let gpu_scan_efficient = self.gpu_scan_efficient.read().unwrap();
        let gpu_stream_compact_efficient = self.gpu_stream_compact_efficient.read().unwrap();
        let gpu_scan_thrust = self.gpu_scan_thrust.read().unwrap();
        let cpu_scan = self.cpu_scan.read().unwrap();
        
        egui::CentralPanel::default().show(ctx, |ui| {
            egui_plot::Plot::new("Plot")
                .allow_drag(false)
                .allow_zoom(false)
                .auto_bounds(Vec2b {
                    x: true, y: true
                })
                .include_x(0.0)
                .include_y(0.0)
                // .include_x(0.0)
                // .include_x(500_00.0)
                // .include_y(0.0)
                // .include_y(600.0)
                .y_axis_label("Algorithm Runtime (ms)")
                .x_axis_label("Array Size (1 << x)")
                .legend(Legend::default().position(Corner::LeftTop))
                .show(ui, |plot_ui|  {
                    match self.current {
                        Tab::Scan => {
                            plot_ui.line(
                                Line::new(gpu_scan_naive.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::RED).name("GPU Scan Naive")
                            );

                            plot_ui.line(
                                Line::new(gpu_scan_efficient.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::BLUE).name("GPU Scan Efficient")
                            );

                            plot_ui.line(
                                Line::new(gpu_scan_thrust.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::GREEN).name("GPU Scan Thrust")
                            );

                            plot_ui.line(
                                Line::new(cpu_scan.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::ORANGE).name("CPU Scan")
                            );
                        },
                        Tab::StreamCompaction => {
                            plot_ui.line(
                                Line::new(gpu_stream_compact_efficient.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::RED).name("GPU Stream Compaction Efficient")
                            );

                            plot_ui.line(
                                Line::new(cpu_stream_compact_with_scan.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::BLUE).name("CPU Stream Compaction With Scan")
                            );

                            plot_ui.line(
                                Line::new(cpu_stream_compact_without_scan.iter().map(|(ms, size)|{
                                    [*size as f64, *ms as f64]
                                }).collect::<Vec<[f64; 2]>>()).color(Color32::GREEN).name("CPU Stream Compaction Without Scan")
                            );
                        }
                    }

                    
                });
        });

        ctx.request_repaint();    
    }
}

async fn run_tests(size: u32, block_size: u32) -> (f32, f32, f32, f32, f32, f32, f32) {
    let mut cuda = tokio::process::Command::new("../build/bin/Release/cis5650_stream_compaction_test.exe")
        .current_dir("../")
        .arg("-size")
        .arg(size.to_string())
        .arg("-blocksize")
        .arg(block_size.to_string())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to start process");

    let stdout = cuda.stdout.take().expect("No stdout captured");
    let mut reader = tokio::io::BufReader::new(stdout).lines();

    let mut cpu_stream_compact_with_scan: f32 = -1.0;
    let mut cpu_stream_compact_without_scan: f32 = -1.0;
    let mut gpu_scan_naive: f32 = -1.0;
    let mut gpu_scan_efficient: f32 = -1.0;
    let mut gpu_stream_compact_efficient: f32 = -1.0;
    let mut gpu_scan_thrust: f32 = -1.0;
    let mut cpu_scan: f32 = -1.0;

    // Spawn task to print stdout
    while let Ok(Some(line)) = reader.next_line().await {
        if line.starts_with("ERROR") {
            println!("{}", line);
        }

        let re = Regex::new(r"^([^:]+):.*?([0-9.]+)ms").unwrap();
        if let Some(caps) = re.captures(&line) {
            match &caps[1] {
                "CPU Compaction with Scan" => cpu_stream_compact_with_scan = caps[2].parse().unwrap(),
                "CPU Compaction without Scan" => cpu_stream_compact_without_scan = caps[2].parse().unwrap(),
                "CPU Scan" => cpu_scan = caps[2].parse().unwrap(),
                "GPU Scan Naive" => gpu_scan_naive = caps[2].parse().unwrap(),
                "GPU Scan Work Efficient" => gpu_scan_efficient = caps[2].parse().unwrap(),
                "GPU Stream Compaction Work Efficient" => gpu_stream_compact_efficient = caps[2].parse().unwrap(),
                "GPU Scan Thrust" => gpu_scan_thrust = caps[2].parse().unwrap(),
                _ => println!("Unaccounted for string: {}", &caps[1])
            }
        } else {
            println!("{} not captured", line);
        }
    }

    return (cpu_stream_compact_with_scan, cpu_stream_compact_without_scan, gpu_scan_naive, gpu_scan_efficient, gpu_stream_compact_efficient, gpu_scan_thrust, cpu_scan);
}
