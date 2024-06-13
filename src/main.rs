use tch::nn::{Module, OptimizerConfig};
use tch::vision::imagenet;
use tch::{nn, vision, Device, Kind, Tensor};

fn main() {
    // Download images of birds and drones from a kaggle dataset
    // https://www.kaggle.com/datasets/harshwalia/birds-vs-drone-dataset

    // Set up GPU
    let device = tch::Device::cuda_if_available();

    // Load the pretrianed ResNet18 model
    let mut vs = nn::VarStore::new(device);
    let net = vision::resnet::resnet18(&vs.root(), tch::vision::imagenet::CLASS_COUNT);
    vs.load(std::path::Path::new("./weights/resnet18.ot"))
        .unwrap();

    let train_dir = "dataset/train";
    let val_dir = "dataset/val";
    let batch_size = 32;
    let train_loader = imagenet::load_from_dir(std::path::Path::new("./dataset"));
    dbg!(train_loader);
    todo!()
}
