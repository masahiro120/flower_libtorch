#include "iostream"
#include <string>
#include <vector>
#include <torch/torch.h>
#include "pointnet_model.h"
#include "input_transform.h"

PointnetImpl::PointnetImpl() : input_transform(3), feature_transform(64) {};
PointnetImpl::PointnetImpl(int num_classes) : input_transform(3), feature_transform(64) {
    this->num_classes = num_classes;
    conv_1 = register_module("conv_1", torch::nn::Conv1d(torch::nn::Conv1dOptions(3, 64, 1)
                .padding(0).stride(1).dilation(1).bias(true)));

    conv_2 = register_module("conv_2", torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 64, 1)
                .padding(0).stride(1).dilation(1).bias(true)));

    conv_3 = register_module("conv_3", torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 1)
                .padding(0).stride(1).dilation(1).bias(true)));

    conv_4 = register_module("conv_4", torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 1024, 1)
                .padding(0).stride(1).dilation(1).bias(true)));

    fc_1 = register_module("fc_1", torch::nn::Linear(1024, 512));
    fc_2 = register_module("fc_2", torch::nn::Linear(512, 256));
    last = register_module("last", torch::nn::Linear(256, num_classes));

    bn_1 = register_module("bn_1", torch::nn::BatchNorm1d(64));
    bn_2 = register_module("bn_2", torch::nn::BatchNorm1d(64));
    bn_3 = register_module("bn_3", torch::nn::BatchNorm1d(128));
    bn_4 = register_module("bn_4", torch::nn::BatchNorm1d(1024));
    bn_5 = register_module("bn_5", torch::nn::BatchNorm1d(512));
    bn_6 = register_module("bn_6", torch::nn::BatchNorm1d(256));

}

at::Tensor PointnetImpl::forward(torch::Tensor x){
    // 時間計測
    auto start = std::chrono::system_clock::now();
    //input --> 1,3,2048
    //InputTransform input_transform (3);
    //input_transform->to(torch::kCPU);
    auto transform = input_transform->forward(x);
    x = torch::matmul(transform,x);
    x = torch::relu(bn_1->forward(conv_1->forward(x)));
    x = torch::relu(bn_2->forward(conv_2->forward(x)));
    //FeatureTransform feature_transform(64);
    //feature_transform->to(torch::kCPU);
    auto features = feature_transform->forward(x);
    x = torch::matmul(features,x);
    x = torch::relu(bn_3->forward(conv_3->forward(x)));
    x = torch::relu(bn_4->forward(conv_4->forward(x)));
    //1, 1024, 2048 --> 1, 1024
    // std::cout<<"before shape is: "<<x.sizes()<<std::endl;
    x = torch::max_pool1d(x,2048,2,0);
    // std::cout<<"after shape is: "<<x.sizes()<<std::endl;
    x = x.view({1024});
    x = fc_1->forward(x);
    x = fc_2->forward(x);
    x = last->forward(x);
    x = x.view({-1,this->num_classes});
    auto end = std::chrono::system_clock::now();
    auto dur = end - start;
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    // microseconds
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
    // std::cout << "forward time is: " << msec << " milli sec" << std::endl;
    // std::cout << "forward time is: " << usec << " micro sec" << std::endl;

    return x;
}

