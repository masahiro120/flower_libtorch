// #include "tiny_dnn_client.h"
// #include "../../../tiny-dnn/tiny_dnn/tiny_dnn.h"
// #include "../../../tiny-dnn/examples/mnist/train.cpp"
// #include "../../../tiny-dnn/tiny_dnn/layers/layer.h"
// #include "../../../tiny-dnn/tiny_dnn/util/weight_init.h"

#include "libtorch_client.h"

using namespace std;

bool check_first_round = false;

// Compare files
    extern float test_data_params_bn_1_bias[64];
    extern float test_data_params_bn_1_weight[64];
    extern float test_data_params_bn_2_bias[64];
    extern float test_data_params_bn_2_weight[64];
    extern float test_data_params_bn_3_bias[128];
    extern float test_data_params_bn_3_weight[128];
    extern float test_data_params_bn_4_bias[1024];
    extern float test_data_params_bn_4_weight[1024];
    extern float test_data_params_bn_5_bias[512];
    extern float test_data_params_bn_5_weight[512];
    extern float test_data_params_bn_6_bias[256];
    extern float test_data_params_bn_6_weight[256];
    extern float test_data_params_conv_1_bias[64];
    extern float test_data_params_conv_1_weight[192];
    extern float test_data_params_conv_2_bias[64];
    extern float test_data_params_conv_2_weight[4096];
    extern float test_data_params_conv_3_bias[128];
    extern float test_data_params_conv_3_weight[8192];
    extern float test_data_params_conv_4_bias[1024];
    extern float test_data_params_conv_4_weight[131072];
    extern float test_data_params_fc_1_bias[512];
    extern float test_data_params_fc_1_weight[524288];
    extern float test_data_params_fc_2_bias[256];
    extern float test_data_params_fc_2_weight[131072];
    extern float test_data_params_last_bias[40];
    extern float test_data_params_last_weight[10240];
    extern float test_data_params_input_transform_bn_1_bias[64];
    extern float test_data_params_input_transform_bn_1_weight[64];
    extern float test_data_params_input_transform_bn_2_bias[128];
    extern float test_data_params_input_transform_bn_2_weight[128];
    extern float test_data_params_input_transform_bn_3_bias[1024];
    extern float test_data_params_input_transform_bn_3_weight[1024];
    extern float test_data_params_input_transform_conv_1_bias[64];
    extern float test_data_params_input_transform_conv_1_weight[192];
    extern float test_data_params_input_transform_conv_2_bias[128];
    extern float test_data_params_input_transform_conv_2_weight[8192];
    extern float test_data_params_input_transform_conv_3_bias[1024];
    extern float test_data_params_input_transform_conv_3_weight[131072];
    extern float test_data_params_input_transform_fc_1_bias[512];
    extern float test_data_params_input_transform_fc_1_weight[524288];
    extern float test_data_params_input_transform_fc_2_bias[256];
    extern float test_data_params_input_transform_fc_2_weight[131072];
    extern float test_data_params_feature_transform_bn_1_bias[64];
    extern float test_data_params_feature_transform_bn_1_weight[64];
    extern float test_data_params_feature_transform_bn_2_bias[128];
    extern float test_data_params_feature_transform_bn_2_weight[128];
    extern float test_data_params_feature_transform_bn_3_bias[1024];
    extern float test_data_params_feature_transform_bn_3_weight[1024];
    extern float test_data_params_feature_transform_conv_1_bias[64];
    extern float test_data_params_feature_transform_conv_1_weight[4096];
    extern float test_data_params_feature_transform_conv_2_bias[128];
    extern float test_data_params_feature_transform_conv_2_weight[8192];
    extern float test_data_params_feature_transform_conv_3_bias[1024];
    extern float test_data_params_feature_transform_conv_3_weight[131072];
    extern float test_data_params_feature_transform_fc_1_bias[512];
    extern float test_data_params_feature_transform_fc_1_weight[524288];
    extern float test_data_params_feature_transform_fc_2_bias[256];
    extern float test_data_params_feature_transform_fc_2_weight[131072];


Libtorch_Client::Libtorch_Client(std::string client_id,
                                 Pointnet &model,
                                 std::vector<int64_t> &train_label,
                                 std::vector<int64_t> &test_label,
                                 std::vector<float> &train_data,
                                 std::vector<float> &test_data,
                                 std::vector<int64_t> &default_test_label,
                                 std::vector<float> &default_test_data)
            : model(model),
              train_label(train_label),
              train_data(train_data),
              test_label(test_label),
              test_data(test_data),
              default_test_label(default_test_label),
              default_test_data(default_test_data) {
};

torch::Device device(torch::kCPU);

extern int DATA_SIZE;

template <class T>
void get_named_parameter(T& model, const char* key, float* layer_name) {
    float* src = model->named_parameters()[key].contiguous().template data_ptr<float>();
    size_t count = model->named_parameters()[key].numel();

    // std::cout << src[count - 1] << " : " << key << std::endl;

    memcpy(layer_name, src, count * sizeof(float));
}

void writeFloatArrayAsDouble(std::ostringstream& oss, const float* array, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        double temp = static_cast<double>(array[i]);
        oss.write(reinterpret_cast<const char*>(&temp), sizeof(double));
    }
}

template <class T>
void set_named_parameter(T& model, const char* key, const float* data) {
    float* dest = model->named_parameters()[key].contiguous().template data_ptr<float>();
    size_t count = model->named_parameters()[key].numel();

    memcpy(dest, data, count * sizeof(float));
}

// プログレスバーを表示する関数
void printProgressBar(int step, int total) {
    const int barWidth = 70;
    // ここでのprogress計算を修正
    float progress = (float)(step + 1) / total;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

/**
 * Return the current local model parameters
 * Simple string are used for now to test communication, needs updates in the future
 */
flwr::ParametersRes Libtorch_Client::get_parameters() {

    std::cout <<"get_parameters Test" << std::endl;
    // Serialize

    // get all parameters
        get_named_parameter(this->model, "bn_1.bias", test_data_params_bn_1_bias);
        get_named_parameter(this->model, "bn_1.weight", test_data_params_bn_1_weight);
        get_named_parameter(this->model, "bn_2.bias", test_data_params_bn_2_bias);
        get_named_parameter(this->model, "bn_2.weight", test_data_params_bn_2_weight);
        get_named_parameter(this->model, "bn_3.bias", test_data_params_bn_3_bias);
        get_named_parameter(this->model, "bn_3.weight", test_data_params_bn_3_weight);
        get_named_parameter(this->model, "bn_4.bias", test_data_params_bn_4_bias);
        get_named_parameter(this->model, "bn_4.weight", test_data_params_bn_4_weight);
        get_named_parameter(this->model, "bn_5.bias", test_data_params_bn_5_bias);
        get_named_parameter(this->model, "bn_5.weight", test_data_params_bn_5_weight);
        get_named_parameter(this->model, "bn_6.bias", test_data_params_bn_6_bias);
        get_named_parameter(this->model, "bn_6.weight", test_data_params_bn_6_weight);
        get_named_parameter(this->model, "conv_1.bias", test_data_params_conv_1_bias);
        get_named_parameter(this->model, "conv_1.weight", test_data_params_conv_1_weight);
        get_named_parameter(this->model, "conv_2.bias", test_data_params_conv_2_bias);
        get_named_parameter(this->model, "conv_2.weight", test_data_params_conv_2_weight);
        get_named_parameter(this->model, "conv_3.bias", test_data_params_conv_3_bias);
        get_named_parameter(this->model, "conv_3.weight", test_data_params_conv_3_weight);
        get_named_parameter(this->model, "conv_4.bias", test_data_params_conv_4_bias);
        get_named_parameter(this->model, "conv_4.weight", test_data_params_conv_4_weight);
        get_named_parameter(this->model, "fc_1.bias", test_data_params_fc_1_bias);
        get_named_parameter(this->model, "fc_1.weight", test_data_params_fc_1_weight);
        get_named_parameter(this->model, "fc_2.bias", test_data_params_fc_2_bias);
        get_named_parameter(this->model, "fc_2.weight", test_data_params_fc_2_weight);
        get_named_parameter(this->model, "last.bias", test_data_params_last_bias);
        get_named_parameter(this->model, "last.weight", test_data_params_last_weight);
        get_named_parameter(this->model->input_transform, "bn_1.bias", test_data_params_input_transform_bn_1_bias);
        get_named_parameter(this->model->input_transform, "bn_1.weight", test_data_params_input_transform_bn_1_weight);
        get_named_parameter(this->model->input_transform, "bn_2.bias", test_data_params_input_transform_bn_2_bias);
        get_named_parameter(this->model->input_transform, "bn_2.weight", test_data_params_input_transform_bn_2_weight);
        get_named_parameter(this->model->input_transform, "bn_3.bias", test_data_params_input_transform_bn_3_bias);
        get_named_parameter(this->model->input_transform, "bn_3.weight", test_data_params_input_transform_bn_3_weight);
        get_named_parameter(this->model->input_transform, "conv_1.bias", test_data_params_input_transform_conv_1_bias);
        get_named_parameter(this->model->input_transform, "conv_1.weight", test_data_params_input_transform_conv_1_weight);
        get_named_parameter(this->model->input_transform, "conv_2.bias", test_data_params_input_transform_conv_2_bias);
        get_named_parameter(this->model->input_transform, "conv_2.weight", test_data_params_input_transform_conv_2_weight);
        get_named_parameter(this->model->input_transform, "conv_3.bias", test_data_params_input_transform_conv_3_bias);
        get_named_parameter(this->model->input_transform, "conv_3.weight", test_data_params_input_transform_conv_3_weight);
        get_named_parameter(this->model->input_transform, "fc_1.bias", test_data_params_input_transform_fc_1_bias);
        get_named_parameter(this->model->input_transform, "fc_1.weight", test_data_params_input_transform_fc_1_weight);
        get_named_parameter(this->model->input_transform, "fc_2.bias", test_data_params_input_transform_fc_2_bias);
        get_named_parameter(this->model->input_transform, "fc_2.weight", test_data_params_input_transform_fc_2_weight);
        get_named_parameter(this->model->feature_transform, "bn_1.bias", test_data_params_feature_transform_bn_1_bias);
        get_named_parameter(this->model->feature_transform, "bn_1.weight", test_data_params_feature_transform_bn_1_weight);
        get_named_parameter(this->model->feature_transform, "bn_2.bias", test_data_params_feature_transform_bn_2_bias);
        get_named_parameter(this->model->feature_transform, "bn_2.weight", test_data_params_feature_transform_bn_2_weight);
        get_named_parameter(this->model->feature_transform, "bn_3.bias", test_data_params_feature_transform_bn_3_bias);
        get_named_parameter(this->model->feature_transform, "bn_3.weight", test_data_params_feature_transform_bn_3_weight);
        get_named_parameter(this->model->feature_transform, "conv_1.bias", test_data_params_feature_transform_conv_1_bias);
        get_named_parameter(this->model->feature_transform, "conv_1.weight", test_data_params_feature_transform_conv_1_weight);
        get_named_parameter(this->model->feature_transform, "conv_2.bias", test_data_params_feature_transform_conv_2_bias);
        get_named_parameter(this->model->feature_transform, "conv_2.weight", test_data_params_feature_transform_conv_2_weight);
        get_named_parameter(this->model->feature_transform, "conv_3.bias", test_data_params_feature_transform_conv_3_bias);
        get_named_parameter(this->model->feature_transform, "conv_3.weight", test_data_params_feature_transform_conv_3_weight);
        get_named_parameter(this->model->feature_transform, "fc_1.bias", test_data_params_feature_transform_fc_1_bias);
        get_named_parameter(this->model->feature_transform, "fc_1.weight", test_data_params_feature_transform_fc_1_weight);
        get_named_parameter(this->model->feature_transform, "fc_2.bias", test_data_params_feature_transform_fc_2_bias);
        get_named_parameter(this->model->feature_transform, "fc_2.weight", test_data_params_feature_transform_fc_2_weight);    

    std::ostringstream oss;
        writeFloatArrayAsDouble(oss, test_data_params_bn_1_bias, sizeof(test_data_params_bn_1_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_1_weight, sizeof(test_data_params_bn_1_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_2_bias, sizeof(test_data_params_bn_2_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_2_weight, sizeof(test_data_params_bn_2_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_3_bias, sizeof(test_data_params_bn_3_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_3_weight, sizeof(test_data_params_bn_3_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_4_bias, sizeof(test_data_params_bn_4_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_4_weight, sizeof(test_data_params_bn_4_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_5_bias, sizeof(test_data_params_bn_5_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_5_weight, sizeof(test_data_params_bn_5_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_6_bias, sizeof(test_data_params_bn_6_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_bn_6_weight, sizeof(test_data_params_bn_6_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_conv_1_bias, sizeof(test_data_params_conv_1_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_conv_1_weight, sizeof(test_data_params_conv_1_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_conv_2_bias, sizeof(test_data_params_conv_2_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_conv_2_weight, sizeof(test_data_params_conv_2_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_conv_3_bias, sizeof(test_data_params_conv_3_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_conv_3_weight, sizeof(test_data_params_conv_3_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_conv_4_bias, sizeof(test_data_params_conv_4_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_conv_4_weight, sizeof(test_data_params_conv_4_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_fc_1_bias, sizeof(test_data_params_fc_1_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_fc_1_weight, sizeof(test_data_params_fc_1_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_fc_2_bias, sizeof(test_data_params_fc_2_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_fc_2_weight, sizeof(test_data_params_fc_2_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_last_bias, sizeof(test_data_params_last_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_last_weight, sizeof(test_data_params_last_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_bn_1_bias, sizeof(test_data_params_input_transform_bn_1_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_bn_1_weight, sizeof(test_data_params_input_transform_bn_1_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_bn_2_bias, sizeof(test_data_params_input_transform_bn_2_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_bn_2_weight, sizeof(test_data_params_input_transform_bn_2_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_bn_3_bias, sizeof(test_data_params_input_transform_bn_3_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_bn_3_weight, sizeof(test_data_params_input_transform_bn_3_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_conv_1_bias, sizeof(test_data_params_input_transform_conv_1_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_conv_1_weight, sizeof(test_data_params_input_transform_conv_1_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_conv_2_bias, sizeof(test_data_params_input_transform_conv_2_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_conv_2_weight, sizeof(test_data_params_input_transform_conv_2_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_conv_3_bias, sizeof(test_data_params_input_transform_conv_3_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_conv_3_weight, sizeof(test_data_params_input_transform_conv_3_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_fc_1_bias, sizeof(test_data_params_input_transform_fc_1_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_fc_1_weight, sizeof(test_data_params_input_transform_fc_1_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_fc_2_bias, sizeof(test_data_params_input_transform_fc_2_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_input_transform_fc_2_weight, sizeof(test_data_params_input_transform_fc_2_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_bn_1_bias, sizeof(test_data_params_feature_transform_bn_1_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_bn_1_weight, sizeof(test_data_params_feature_transform_bn_1_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_bn_2_bias, sizeof(test_data_params_feature_transform_bn_2_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_bn_2_weight, sizeof(test_data_params_feature_transform_bn_2_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_bn_3_bias, sizeof(test_data_params_feature_transform_bn_3_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_bn_3_weight, sizeof(test_data_params_feature_transform_bn_3_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_conv_1_bias, sizeof(test_data_params_feature_transform_conv_1_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_conv_1_weight, sizeof(test_data_params_feature_transform_conv_1_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_conv_2_bias, sizeof(test_data_params_feature_transform_conv_2_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_conv_2_weight, sizeof(test_data_params_feature_transform_conv_2_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_conv_3_bias, sizeof(test_data_params_feature_transform_conv_3_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_conv_3_weight, sizeof(test_data_params_feature_transform_conv_3_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_fc_1_bias, sizeof(test_data_params_feature_transform_fc_1_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_fc_1_weight, sizeof(test_data_params_feature_transform_fc_1_weight) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_fc_2_bias, sizeof(test_data_params_feature_transform_fc_2_bias) / sizeof(float));
        writeFloatArrayAsDouble(oss, test_data_params_feature_transform_fc_2_weight, sizeof(test_data_params_feature_transform_fc_2_weight) / sizeof(float));
    
    std::list<std::string> tensors;
    tensors.push_back(oss.str());

    std::string tensor_str = "cpp_double";

    std::cout <<"get_parameters Test end" << std::endl;

    return flwr::Parameters(tensors, tensor_str);
};

void Libtorch_Client::set_parameters(flwr::Parameters params) {

    std::cout <<"set_parameters Test" << std::endl;

    std::list<std::string> s = params.getTensors();
    std::cout << "Received " << s.size() <<" Layers from server:" << std::endl;
    // リストが空でないことを確認
    if (s.size() != 0) {
        // 最初の要素へのアクセス
        auto it = s.begin();
        std::cout << "First element size: " << it->size() / sizeof(double) << std::endl; // 最初の要素のサイズ

        // 2番目の要素へのアクセス（リストが2つ以上の要素を持っている場合）
        std::advance(it, 1); // イテレータを1つ進める
        if (it != s.end()) {
            std::cout << "Second element content: " << *it << std::endl; // 2番目の要素の内容
        }
    }

    std::vector<int> layer_params_num = {
      64, 64, 64, 64, 128, 128, 1024, 1024, 512, 512, 256, 256, 64, 192, 64, 4096, 128, 8192, 1024, 131072, 512, 524288, 256, 131072, 40, 10240, 
      64, 64, 128, 128, 1024, 1024, 64, 192, 128, 8192, 1024, 131072, 512, 524288, 256, 131072, 
      64, 64, 128, 128, 1024, 1024, 64, 4096, 128, 8192, 1024, 131072, 512, 524288, 256, 131072
    };

    // Check if the list is not empty
    if (s.size() != 0) {
        // Loop through the list of tensors
        auto it = s.begin();
        const std::string& binaryData = *it; // すべてのデータが連結されていると仮定

        // バイナリデータのポインタ
        const char* dataPtr = binaryData.data();

        // 各レイヤーのパラメータ数に基づいてデータを変換し、静的配列を更新
        size_t offset = 0;
        for (size_t i = 0; i < layer_params_num.size(); ++i) {
            float* currentLayer = nullptr;
            if (i == 0)       currentLayer = test_data_params_bn_1_bias;
            else if (i == 1)  currentLayer = test_data_params_bn_1_weight;
            else if (i == 2)  currentLayer = test_data_params_bn_2_bias;
            else if (i == 3)  currentLayer = test_data_params_bn_2_weight;
            else if (i == 4)  currentLayer = test_data_params_bn_3_bias;
            else if (i == 5)  currentLayer = test_data_params_bn_3_weight;
            else if (i == 6)  currentLayer = test_data_params_bn_4_bias;
            else if (i == 7)  currentLayer = test_data_params_bn_4_weight;
            else if (i == 8)  currentLayer = test_data_params_bn_5_bias;
            else if (i == 9)  currentLayer = test_data_params_bn_5_weight;
            else if (i == 10) currentLayer = test_data_params_bn_6_bias;
            else if (i == 11) currentLayer = test_data_params_bn_6_weight;
            else if (i == 12) currentLayer = test_data_params_conv_1_bias;
            else if (i == 13) currentLayer = test_data_params_conv_1_weight;
            else if (i == 14) currentLayer = test_data_params_conv_2_bias;
            else if (i == 15) currentLayer = test_data_params_conv_2_weight;
            else if (i == 16) currentLayer = test_data_params_conv_3_bias;
            else if (i == 17) currentLayer = test_data_params_conv_3_weight;
            else if (i == 18) currentLayer = test_data_params_conv_4_bias;
            else if (i == 19) currentLayer = test_data_params_conv_4_weight;
            else if (i == 20) currentLayer = test_data_params_fc_1_bias;
            else if (i == 21) currentLayer = test_data_params_fc_1_weight;
            else if (i == 22) currentLayer = test_data_params_fc_2_bias;
            else if (i == 23) currentLayer = test_data_params_fc_2_weight;
            else if (i == 24) currentLayer = test_data_params_last_bias;
            else if (i == 25) currentLayer = test_data_params_last_weight;
            else if (i == 26) currentLayer = test_data_params_input_transform_bn_1_bias;
            else if (i == 27) currentLayer = test_data_params_input_transform_bn_1_weight;
            else if (i == 28) currentLayer = test_data_params_input_transform_bn_2_bias;
            else if (i == 29) currentLayer = test_data_params_input_transform_bn_2_weight;
            else if (i == 30) currentLayer = test_data_params_input_transform_bn_3_bias;
            else if (i == 31) currentLayer = test_data_params_input_transform_bn_3_weight;
            else if (i == 32) currentLayer = test_data_params_input_transform_conv_1_bias;
            else if (i == 33) currentLayer = test_data_params_input_transform_conv_1_weight;
            else if (i == 34) currentLayer = test_data_params_input_transform_conv_2_bias;
            else if (i == 35) currentLayer = test_data_params_input_transform_conv_2_weight;
            else if (i == 36) currentLayer = test_data_params_input_transform_conv_3_bias;
            else if (i == 37) currentLayer = test_data_params_input_transform_conv_3_weight;
            else if (i == 38) currentLayer = test_data_params_input_transform_fc_1_bias;
            else if (i == 39) currentLayer = test_data_params_input_transform_fc_1_weight;
            else if (i == 40) currentLayer = test_data_params_input_transform_fc_2_bias;
            else if (i == 41) currentLayer = test_data_params_input_transform_fc_2_weight;
            else if (i == 42) currentLayer = test_data_params_feature_transform_bn_1_bias;
            else if (i == 43) currentLayer = test_data_params_feature_transform_bn_1_weight;
            else if (i == 44) currentLayer = test_data_params_feature_transform_bn_2_bias;
            else if (i == 45) currentLayer = test_data_params_feature_transform_bn_2_weight;
            else if (i == 46) currentLayer = test_data_params_feature_transform_bn_3_bias;
            else if (i == 47) currentLayer = test_data_params_feature_transform_bn_3_weight;
            else if (i == 48) currentLayer = test_data_params_feature_transform_conv_1_bias;
            else if (i == 49) currentLayer = test_data_params_feature_transform_conv_1_weight;
            else if (i == 50) currentLayer = test_data_params_feature_transform_conv_2_bias;
            else if (i == 51) currentLayer = test_data_params_feature_transform_conv_2_weight;
            else if (i == 52) currentLayer = test_data_params_feature_transform_conv_3_bias;
            else if (i == 53) currentLayer = test_data_params_feature_transform_conv_3_weight;
            else if (i == 54) currentLayer = test_data_params_feature_transform_fc_1_bias;
            else if (i == 55) currentLayer = test_data_params_feature_transform_fc_1_weight;
            else if (i == 56) currentLayer = test_data_params_feature_transform_fc_2_bias;
            else if (i == 57) currentLayer = test_data_params_feature_transform_fc_2_weight;

            for (int j = 0; j < layer_params_num[i]; ++j) {
                double tempDouble;
                std::memcpy(&tempDouble, dataPtr + offset, sizeof(double));
                offset += sizeof(double);

                // doubleからfloatへの変換
                currentLayer[j] = static_cast<float>(tempDouble);
            }
        }

    }

    // Overwrite parameters
        set_named_parameter(this->model, "bn_1.bias", test_data_params_bn_1_bias);
        set_named_parameter(this->model, "bn_1.weight", test_data_params_bn_1_weight);
        set_named_parameter(this->model, "bn_2.bias", test_data_params_bn_2_bias);
        set_named_parameter(this->model, "bn_2.weight", test_data_params_bn_2_weight);
        set_named_parameter(this->model, "bn_3.bias", test_data_params_bn_3_bias);
        set_named_parameter(this->model, "bn_3.weight", test_data_params_bn_3_weight);
        set_named_parameter(this->model, "bn_4.bias", test_data_params_bn_4_bias);
        set_named_parameter(this->model, "bn_4.weight", test_data_params_bn_4_weight);
        set_named_parameter(this->model, "bn_5.bias", test_data_params_bn_5_bias);
        set_named_parameter(this->model, "bn_5.weight", test_data_params_bn_5_weight);
        set_named_parameter(this->model, "bn_6.bias", test_data_params_bn_6_bias);
        set_named_parameter(this->model, "bn_6.weight", test_data_params_bn_6_weight);
        set_named_parameter(this->model, "conv_1.bias", test_data_params_conv_1_bias);
        set_named_parameter(this->model, "conv_1.weight", test_data_params_conv_1_weight);
        set_named_parameter(this->model, "conv_2.bias", test_data_params_conv_2_bias);
        set_named_parameter(this->model, "conv_2.weight", test_data_params_conv_2_weight);
        set_named_parameter(this->model, "conv_3.bias", test_data_params_conv_3_bias);
        set_named_parameter(this->model, "conv_3.weight", test_data_params_conv_3_weight);
        set_named_parameter(this->model, "conv_4.bias", test_data_params_conv_4_bias);
        set_named_parameter(this->model, "conv_4.weight", test_data_params_conv_4_weight);
        set_named_parameter(this->model, "fc_1.bias", test_data_params_fc_1_bias);
        set_named_parameter(this->model, "fc_1.weight", test_data_params_fc_1_weight);
        set_named_parameter(this->model, "fc_2.bias", test_data_params_fc_2_bias);
        set_named_parameter(this->model, "fc_2.weight", test_data_params_fc_2_weight);
        set_named_parameter(this->model, "last.bias", test_data_params_last_bias);
        set_named_parameter(this->model, "last.weight", test_data_params_last_weight);
        set_named_parameter(this->model->input_transform, "bn_1.bias", test_data_params_input_transform_bn_1_bias);
        set_named_parameter(this->model->input_transform, "bn_1.weight", test_data_params_input_transform_bn_1_weight);
        set_named_parameter(this->model->input_transform, "bn_2.bias", test_data_params_input_transform_bn_2_bias);
        set_named_parameter(this->model->input_transform, "bn_2.weight", test_data_params_input_transform_bn_2_weight);
        set_named_parameter(this->model->input_transform, "bn_3.bias", test_data_params_input_transform_bn_3_bias);
        set_named_parameter(this->model->input_transform, "bn_3.weight", test_data_params_input_transform_bn_3_weight);
        set_named_parameter(this->model->input_transform, "conv_1.bias", test_data_params_input_transform_conv_1_bias);
        set_named_parameter(this->model->input_transform, "conv_1.weight", test_data_params_input_transform_conv_1_weight);
        set_named_parameter(this->model->input_transform, "conv_2.bias", test_data_params_input_transform_conv_2_bias);
        set_named_parameter(this->model->input_transform, "conv_2.weight", test_data_params_input_transform_conv_2_weight);
        set_named_parameter(this->model->input_transform, "conv_3.bias", test_data_params_input_transform_conv_3_bias);
        set_named_parameter(this->model->input_transform, "conv_3.weight", test_data_params_input_transform_conv_3_weight);
        set_named_parameter(this->model->input_transform, "fc_1.bias", test_data_params_input_transform_fc_1_bias);
        set_named_parameter(this->model->input_transform, "fc_1.weight", test_data_params_input_transform_fc_1_weight);
        set_named_parameter(this->model->input_transform, "fc_2.bias", test_data_params_input_transform_fc_2_bias);
        set_named_parameter(this->model->input_transform, "fc_2.weight", test_data_params_input_transform_fc_2_weight);
        set_named_parameter(this->model->feature_transform, "bn_1.bias", test_data_params_feature_transform_bn_1_bias);
        set_named_parameter(this->model->feature_transform, "bn_1.weight", test_data_params_feature_transform_bn_1_weight);
        set_named_parameter(this->model->feature_transform, "bn_2.bias", test_data_params_feature_transform_bn_2_bias);
        set_named_parameter(this->model->feature_transform, "bn_2.weight", test_data_params_feature_transform_bn_2_weight);
        set_named_parameter(this->model->feature_transform, "bn_3.bias", test_data_params_feature_transform_bn_3_bias);
        set_named_parameter(this->model->feature_transform, "bn_3.weight", test_data_params_feature_transform_bn_3_weight);
        set_named_parameter(this->model->feature_transform, "conv_1.bias", test_data_params_feature_transform_conv_1_bias);
        set_named_parameter(this->model->feature_transform, "conv_1.weight", test_data_params_feature_transform_conv_1_weight);
        set_named_parameter(this->model->feature_transform, "conv_2.bias", test_data_params_feature_transform_conv_2_bias);
        set_named_parameter(this->model->feature_transform, "conv_2.weight", test_data_params_feature_transform_conv_2_weight);
        set_named_parameter(this->model->feature_transform, "conv_3.bias", test_data_params_feature_transform_conv_3_bias);
        set_named_parameter(this->model->feature_transform, "conv_3.weight", test_data_params_feature_transform_conv_3_weight);
        set_named_parameter(this->model->feature_transform, "fc_1.bias", test_data_params_feature_transform_fc_1_bias);
        set_named_parameter(this->model->feature_transform, "fc_1.weight", test_data_params_feature_transform_fc_1_weight);
        set_named_parameter(this->model->feature_transform, "fc_2.bias", test_data_params_feature_transform_fc_2_bias);
        set_named_parameter(this->model->feature_transform, "fc_2.weight", test_data_params_feature_transform_fc_2_weight);

    std::cout <<"set_parameters Test end" << std::endl;
};

flwr::PropertiesRes Libtorch_Client::get_properties(flwr::PropertiesIns ins) {
    
    std::cout <<"get_properties Test" << std::endl;

    flwr::PropertiesRes p;
    p.setPropertiesRes(static_cast<flwr::Properties>(ins.getPropertiesIns()));

    std::cout <<"get_properties Test end" << std::endl;
    return p;
}

/**
 * Refine the provided weights using the locally held dataset
 * Simple settings are used for testing, needs updates in the future
 */
flwr::FitRes Libtorch_Client::fit(flwr::FitIns ins) {
    std::cout << "Fitting..." << std::endl;
    flwr::FitRes resp;

    flwr::Parameters p = ins.getParameters();
    std::list<std::string> p_tensor = p.getTensors();
    this->set_parameters(p);

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(2e-4).betas({0.9, 0.5}));

    int num_epochs = 10;

    this->model->to(device);
    this->model->input_transform->to(device);
    this->model->feature_transform->to(device);

    const int INPUT_COUNT = DATA_SIZE * 0.8;
    const int POINT_COUNT = 2048;

    std::vector<std::vector<size_t>> vec;
    vec.resize(POINT_COUNT);

    std::cout << "start training" << std::endl;


    // Import data and label
    torch::Tensor inpt_tensor = torch::from_blob(train_data.data(), {INPUT_COUNT, POINT_COUNT, 3}, torch::kFloat32);
    torch::Tensor label_tensor = torch::from_blob(train_label.data(), {INPUT_COUNT, 1}, torch::kInt64);

    for(int epoch = 0; epoch < num_epochs; epoch++) {
        for(int btch = 0; btch < INPUT_COUNT; btch++) {
            torch::Tensor inp = inpt_tensor[btch];
            torch::Tensor labl = label_tensor[btch];
            inp = inp.to(torch::kFloat32).to(device);
            labl = labl.to(torch::kInt64).to(device);

            float* data_dest = inp.contiguous().template data_ptr<float>();
            size_t count = inp.numel();

            // std::cout << "Data tensor[" << btch << "]:" << data_dest[0] << ", " << data_dest[1] << ", " << data_dest[2] << ", " << data_dest[3] << ", " << data_dest[4] << ", " << data_dest[5] << std::endl;

            int64_t* label_dest = labl.contiguous().template data_ptr<int64_t>();
            size_t label_count = labl.numel();

            // std::cout << "Label tensor[" << btch << "]:" << label_dest[0] << std::endl;


            inp = inp.view({1, 3, 2048});
            optimizer.zero_grad();
            auto output = this->model->forward(inp);
            auto loss = torch::nll_loss(torch::log_softmax(output, 1), labl);
            loss.backward();
            optimizer.step();
            // std::cout << "Epoch: " << epoch + 1 << " Batch: " << btch << " Loss: " << loss.template item<float>() << std::endl;
            printProgressBar(btch, INPUT_COUNT);
        }
        std::cout << std::endl;

        float Acc = 0.0;
        float Mse = 0.0;
        for(int btch = 0; btch < INPUT_COUNT; btch++) {
            torch::Tensor inp = inpt_tensor[btch];
            torch::Tensor labl = label_tensor[btch];
            inp = inp.to(torch::kFloat32).to(device);
            labl = labl.to(torch::kInt64).to(device);
            inp = inp.view({1, 3, 2048});
            auto output = this->model->forward(inp);
            auto loss = torch::nll_loss(torch::log_softmax(output, 1), labl);
            auto acc = output.argmax(1).eq(labl).sum();
            Acc += acc.template item<float>();
            Mse += loss.template item<float>();
        }

        Acc = Acc / float(INPUT_COUNT);
        Mse = Mse / float(INPUT_COUNT);
        std::cout << "Epoch: " << epoch + 1 << " Mse is: " << Mse << " Accuracy is: " << Acc * 100 << "%" << std::endl;
    }

    size_t train_size = this->train_data.size();

    // std::tuple<size_t, float, float> result = std::make_tuple(train_size, train_accuracy, train_accuracy);

    resp.setParameters(this->get_parameters().getParameters());
    resp.setNum_example(train_size);


    std::cout <<"Fit Test end" << std::endl;

    return resp;
};


/**
 * Evaluate the provided weights using the locally held dataset
 * Needs updates in the future
 */
flwr::EvaluateRes Libtorch_Client::evaluate(flwr::EvaluateIns ins) {
    std::cout << "evaluate Test" << std::endl;

    std::cout << "Evaluating..." << std::endl;
    flwr::EvaluateRes resp;
    flwr::Parameters p = ins.getParameters();
    this->set_parameters(p);

    this->model->to(device);
    this->model->input_transform->to(device);
    this->model->feature_transform->to(device);
    
    const int INPUT_COUNT = DATA_SIZE * 0.2;
    const int POINT_COUNT = 2048;

    torch::Tensor inpt_tensor = torch::from_blob(test_data.data(), {INPUT_COUNT, POINT_COUNT, 3}, torch::kFloat32);

    torch::Tensor label_tensor = torch::from_blob(test_label.data(), {INPUT_COUNT, 1}, torch::kInt64);

    float Acc = 0.0;
    float Loss = 0.0;
    for(int btch = 0; btch < INPUT_COUNT; btch++) {
        torch::Tensor inp = inpt_tensor[btch];
        torch::Tensor labl = label_tensor[btch];
        inp = inp.to(torch::kFloat32).to(device);
        labl = labl.to(torch::kInt64).to(device);
        inp = inp.view({1, 3, 2048});
        auto output = model->forward(inp);
        auto loss = torch::nll_loss(torch::log_softmax(output, 1), labl);
        auto acc = output.argmax(1).eq(labl).sum();
        Acc += acc.template item<float>();
        Loss += loss.template item<float>();
        printProgressBar(btch, INPUT_COUNT);
    }
    std::cout << std::endl;
    Loss = Loss / float(INPUT_COUNT);
    Acc = Acc / float(INPUT_COUNT);
    std::cout << "Test Loss is: " << Loss << " Accuracy is: " << Acc * 100 << "%" << std::endl;
    std::cout << "Testing done" << std::endl;
    
    size_t test_size = INPUT_COUNT;

    std::tuple<size_t, float, float> result = std::make_tuple(test_size, Loss, Loss);

    resp.setNum_example(std::get<0>(result));
    resp.setLoss(std::get<1>(result));

    flwr::Scalar loss_metric = flwr::Scalar();
    loss_metric.setFloat(std::get<2>(result));
    std::map<std::string, flwr::Scalar> metric = {{"loss", loss_metric}};
    resp.setMetrics(metric);

    std::cout <<"evaluate Test end" << std::endl;

    return resp;

};


