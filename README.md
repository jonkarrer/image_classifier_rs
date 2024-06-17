# Image Classifier Using Rust

A walkthrough of a bare bones classifier using Rust, specifically a crate that uses pytorch bindings, `tch-rs` to train and classify images. We will be doing transfer learning with a resnet18 model.

Most of the code comes from this tutorial: <https://github.com/LaurentMazare/tch-rs/tree/main/examples/transfer-learning>

## Setup

```shell
cargo new image_classifier_rust --bin
cd image_classifier_rust
```

- Add `tch-rs` as a dependency, and `anyhow` for error handling

```shell
cargo add tch-rs,anyhow
```

## Gathering Data

Find a data set online, I used <https://www.kaggle.com/datasets/harshwalia/birds-vs-drone-dataset>.

After downloading a data set for the classifier, place it in a `dataset` directory at the root of the project. Make 2 subdirectories for training and testing, called `train` and `val`. In each of these subdirectories, put the images under another subdirectory names according to what it is. This will act as your labels/classes. Put about 10-20% of the images in `val` directory.

Here is an example of the structure:

```shell
dataset
├── train
│   ├── drone (300 images)
│   ├── bird (300 images)
├── val
│   ├── drone (50 images)
│   ├── bird (50 images)
```

We also need the weights for out base model. We will be using the ResNet18 model. Download it and save it to `weights` directory.

[Link to weight download](https://github.com/LaurentMazare/tch-rs/releases/download/mw/resnet18.ot)

## Training

Now we code.

```rust
use anyhow::Result;
use tch::nn::OptimizerConfig;
use tch::vision::imagenet;
use tch::{nn, vision};

fn main() -> Result<()> {
    // Set up GPU
    let device = tch::Device::cuda_if_available();

    // Load the pretrianed ResNet18 model
    let mut vs = nn::VarStore::new(device);
    let net = vision::resnet::resnet18_no_final_layer(&vs.root());
    vs.load(std::path::Path::new("./weights/resnet18.ot"))
        .unwrap();

    // Pre-compute the final activations.
    let dataset = imagenet::load_from_dir(std::path::Path::new("./dataset")).unwrap();
    let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
    let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
    let mut sgd = nn::Sgd::default().build(&vs, 1e-3)?;

    for epoch_idx in 1..1001 {
        let predicted = train_images.apply(&linear);
        let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
        sgd.backward_step(&loss);

        let test_accuracy = test_images
            .apply(&linear)
            .accuracy_for_logits(&dataset.test_labels);
        println!("{} {:.2}%", epoch_idx, 100. * f64::try_from(test_accuracy)?);
    }

    // Save the model
    vs.save(std::path::Path::new("./weights/resnet18_linear.ot"))?;
    println!("Saved weights to ./weights/resnet18_linear.ot");

    Ok(())
}
```

Certainly! Your Rust code using the `tch-rs` library (which provides Rust bindings for the Torch library) is designed to train a simple image classifier. Here’s a step-by-step explanation of what each significant part of the code is doing:

### Code Breakdown

#### Set Up GPU

```rust
let device = tch::Device::cuda_if_available();
```

This line checks if a CUDA-capable GPU is available on the system and uses it if possible. This can significantly accelerate training by performing computations on the GPU rather than the CPU.

#### Load Pretrained ResNet18 Model

```rust
let mut vs = nn::VarStore::new(device);
let net = vision::resnet::resnet18_no_final_layer(&vs.root());
vs.load(std::path::Path::new("./weights/resnet18.ot")).unwrap();
```

Here, a variable store is created on the chosen device, and a pre-trained ResNet18 model (excluding the final layer) is loaded. The `vs.load` function loads the pretrained weights into this model from a file.

#### Pre-compute the Final Activations

```rust
let dataset = imagenet::load_from_dir(std::path::Path::new("./dataset")).unwrap();
let train_images = tch::no_grad(|| dataset.train_images.apply_t(&net, false));
let test_images = tch::no_grad(|| dataset.test_images.apply_t(&net, false));

```

The dataset is loaded from a directory, and then the training and testing images are passed through the ResNet18 model to get their embeddings (feature vectors). This step essentially transforms the raw images into a form that's easier to classify using a simple linear model.

#### Define the Classifier Layer

```rust
let vs = nn::VarStore::new(tch::Device::Cpu);
let linear = nn::linear(vs.root(), 512, dataset.labels, Default::default());
let mut sgd = nn::Sgd::default().build(&vs, 1e-3)?;
```

After computing the image embeddings, a new variable store is initialized for the classifier (a linear layer) which will operate on these embeddings. The `nn::linear` layer is a simple fully connected layer that maps the 512-dimensional ResNet output to the number of labels (categories) in the dataset. Stochastic Gradient Descent (SGD) is used as the optimizer.

#### Training Loop

```rust
for epoch_idx in 1..1001 {
    let predicted = train_images.apply(&linear);
    let loss = predicted.cross_entropy_for_logits(&dataset.train_labels);
    sgd.backward_step(&loss);

    let test_accuracy = test_images
        .apply(&linear)
        .accuracy_for_logits(&dataset.test_labels);
    println!("{} {:.2}%", epoch_idx, 100. * f64::try_from(test_accuracy)?);
}
```

The training loop runs for 1000 epochs. In each epoch, the linear layer's output is computed for the training images, and the cross-entropy loss is calculated. The loss is then used to update the model weights via back propagation. After updating the weights, the accuracy on the test set is computed and printed.

#### Save the Model

```rust
vs.save(std::path::Path::new("./weights/resno18_linear.ot"))?;
println!("Saved weights to ./weights/resnet18_linear.ot");
```

After training is complete, the weights of the linear layer are saved to disk.

### Use the Model

Now we can use the newly trained model. Some code.

```rust
use anyhow::Result;
use tch::{
    nn::{self, FuncT, Linear, Module, ModuleT},
    vision::{self, imagenet},
    Device, Kind, Tensor,
};

fn load_resnet_no_final_layer(weights_path: &str, device: Device) -> FuncT<'static> {
    let mut vs = nn::VarStore::new(device);
    let net = vision::resnet::resnet18_no_final_layer(&vs.root());
    vs.load(std::path::Path::new(weights_path))
        .expect("Failed to load resnet weights");

    println!("Loaded resnet18 model from {}", weights_path);
    net
}

fn load_trained_layer(weights_path: &str, device: Device) -> Linear {
    let mut vs = nn::VarStore::new(device);
    let linear = nn::linear(vs.root(), 512, 2, Default::default());
    vs.load(weights_path)
        .expect("Failed to load trained weights");

    println!("Loaded linear model from {}", weights_path);
    linear
}

fn process_test_image(image_path: &str, device: Device) -> Tensor {
    imagenet::load_image_and_resize224(image_path)
        .expect("Failed to load image")
        .unsqueeze(0) // Add batch dimension
        .to_device(device) // Make sure it's on the GPU
}

pub fn run_test_on_image(image_path: &str) -> Result<()> {
    let device = tch::Device::cuda_if_available();
    let test_image = process_test_image(image_path, device);

    // Pass image through the base resnet model
    let resnet_features = tch::no_grad(|| {
        load_resnet_no_final_layer("weights/resnet18.ot", device).forward_t(&test_image, false)
    });

    // Pass the resnet features through the linear model
    let logits = tch::no_grad(|| {
        load_trained_layer("weights/resnet18_linear.ot", device).forward(&resnet_features)
    });

    // Get the top 2 predictions
    let labels = vec!["drone", "bird"];
    let output = logits.softmax(-1, Kind::Float);
    let (top_probs, top_idxs) = output.topk(2, -1, true, true);

    println!("I think..:");
    for i in 0..2 {
        let prob = top_probs.double_value(&[0, i]);
        let idx = top_idxs.int64_value(&[0, i]) as usize;
        if let Some(class_name) = labels.get(idx) {
            println!("{:50} {:5.2}%", class_name, 100.0 * prob);
        }
    }

    Ok(())
}

```

### Summary

The code trains a classifier to distinguish between birds and drones using transfer learning, where the initial feature extraction is performed by a pre-trained ResNet18 model. Only the final classification layer is trained from scratch. This is a common technique in image processing tasks to leverage existing high-performing models and adapt them to specific, possibly smaller, datasets.
