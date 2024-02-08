#include <algorithm>
#include <random>

#include "tiny_dnn_client.h"
#include "start.h"
#include "../../../tiny-dnn/tiny_dnn/tiny_dnn.h"
#include "../../../tiny-dnn/tiny_dnn/io/layer_factory.h"

#include "../gray_train_images.cpp"
#include "../gray_train_labels.cpp"

// #include "half_function.cc"

// extern std::vector<tiny_dnn::vec_t> train_images_data;
// extern std::vector<tiny_dnn::label_t> train_labels_data;

extern float train_images_data[80*43][45*45];
extern int train_labels_data[80*43];

#define O true
#define X false

static const bool tbl[] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Client takes three arguments as follows: " << std::endl;
        std::cout << "./client  CLIENT_ID  SERVER_URL" << std::endl;
        std::cout << "Example: ./flwr_client 0 '[::]:8888'" << std::endl;
        return 0;
    }

    // Parsing arguments
    const std::string CLIENT_ID = argv[1];
    const std::string SERVER_URL = argv[2];

    int classes = 43;
    int data_per_class = 80;

    // シャッフル
    std::vector<tiny_dnn::vec_t> train_images(data_per_class * classes / 2 * 0.8);
    std::vector<tiny_dnn::vec_t> test_images(data_per_class * classes / 2 * 0.2);
    std::vector<tiny_dnn::label_t> train_labels(data_per_class * classes / 2 * 0.8);
    std::vector<tiny_dnn::label_t> test_labels(data_per_class * classes / 2 * 0.2);
    // std::random_device seed_gen;
    // std::mt19937 engine(seed_gen());

    const unsigned int seed = 123;
    std::mt19937 engine(seed);

    std::vector<int> num_list;

    for (int i=0; i< data_per_class * classes; i++) num_list.push_back(i);
    std::shuffle(num_list.begin(), num_list.end(), engine);

    // if(CLIENT_ID == "0") {
    //   printf("Client 0\n");
    //   for(int i = 0;i < train_images_data.size() / 2;i++){
    //     if (i % 5 == 0){
    //       // test
    //       test_images.push_back(train_images_data[num_list[i]]);
    //       test_labels.push_back(train_labels_data[num_list[i]]);
    //     } else {
    //       // train
    //       train_images.push_back(train_images_data[num_list[i]]);
    //       train_labels.push_back(train_labels_data[num_list[i]]);
    //     }
    //   }
    // } else {
    //   printf("Client 1\n");
    //   for(int i = train_images_data.size() / 2;i < train_images_data.size();i++){
    //     if (i % 5 == 0){
    //       // test
    //       test_images.push_back(train_images_data[num_list[i]]);
    //       test_labels.push_back(train_labels_data[num_list[i]]);
    //     } else {
    //       // train
    //       train_images.push_back(train_images_data[num_list[i]]);
    //       train_labels.push_back(train_labels_data[num_list[i]]);
    //     }
    //   }
    // }
    int test_index = 0;
    int train_index = 0;

    if(CLIENT_ID == "0") {
      printf("Client 0\n");
      for(int i = 0; i < data_per_class * classes / 2; i++){
        // printf("i: %d\n", i);
        // printf("num_list[i]: %d\n", num_list[i]);
        if (i % 5 == 0){
          // test
          test_images[test_index].resize(45 * 45);
          for(int j = 0; j < 45 * 45; j++){
            test_images[test_index][j] = train_images_data[num_list[i]][j];
          }
          test_labels[test_index] = train_labels_data[num_list[i]];
          test_index++;
        } else {
          // train
          train_images[train_index].resize(45 * 45);
          for(int j = 0; j < 45 * 45; j++){
            train_images[train_index][j] = train_images_data[num_list[i]][j];
          }
          train_labels[train_index] = train_labels_data[num_list[i]];
          train_index++;
        }
      }
    } else {
      printf("Client 1\n");
      for(int i = data_per_class * classes / 2; i < data_per_class * classes; i++){
        if (i % 5 == 0){
          // test
          test_images[test_index].resize(45 * 45);
          for(int j = 0; j < 45 * 45; j++){
            test_images[test_index][j] = train_images_data[num_list[i]][j];
          }
          test_labels[test_index] = train_labels_data[num_list[i]];
          test_index++;
        } else {
          // train
          train_images[train_index].resize(45 * 45);
          for(int j = 0; j < 45 * 45; j++){
            train_images[train_index][j] = train_images_data[num_list[i]][j];
          }
          train_labels[train_index] = train_labels_data[num_list[i]];
          train_index++;
        }
      }
    }
    
    
    // for(int i = 0;i < 20 * 43;i++){
    //   if (i % 5 == 0){
    //     // test
    //     test_images.push_back(train_images_data[num_list[i]]);
    //     test_labels.push_back(train_labels_data[num_list[i]]);
    //   } else {
    //     // train
    //     train_images.push_back(train_images_data[num_list[i]]);
    //     train_labels.push_back(train_labels_data[num_list[i]]);
    //   }

    // }

    // std::vector<tiny_dnn::vec_t> client_train_images;
    // std::vector<tiny_dnn::label_t> client_train_labels;
    // std::vector<tiny_dnn::vec_t> client_test_images;
    // std::vector<tiny_dnn::label_t> client_test_labels;

    // if(CLIENT_ID == "1") {
    //   std::vector<tiny_dnn::vec_t> half_train_images(train_images.begin(), train_images.begin() + train_images.size() / 2);
    //   std::vector<tiny_dnn::label_t> half_train_labels(train_labels.begin(), train_labels.begin() + train_labels.size() / 2);
    //   std::vector<tiny_dnn::vec_t> half_test_images(test_images.begin(), test_images.begin() + test_images.size() / 2);
    //   std::vector<tiny_dnn::label_t> half_test_labels(test_labels.begin(), test_labels.begin() + test_labels.size() / 2);

    //   client_train_images = half_train_images;
    //   client_train_labels = half_train_labels;
    //   client_test_images = half_test_images;
    //   client_test_labels = half_test_labels;
    // } else {
    //   std::vector<tiny_dnn::vec_t> half_train_images(train_images.begin() + train_images.size() / 2, train_images.end());
    //   std::vector<tiny_dnn::label_t> half_train_labels(train_labels.begin() + train_labels.size() / 2, train_labels.end());
    //   std::vector<tiny_dnn::vec_t> half_test_images(test_images.begin() + test_images.size() / 2, test_images.end());
    //   std::vector<tiny_dnn::label_t> half_test_labels(test_labels.begin() + test_labels.size() / 2, test_labels.end());

    //   client_train_images = half_train_images;
    //   client_train_labels = half_train_labels;
    //   client_test_images = half_test_images;
    //   client_test_labels = half_test_labels;
    // }

    std::cout << "Train data images size : " << train_images.size() << std::endl;
    std::cout << "Train data labels size : " << train_labels.size() << std::endl;



    // Define a model
    // LineFitModel model = LineFitModel(500, 0.01, ms.size());
    using namespace tiny_dnn::layers;

    using conv = tiny_dnn::convolutional_layer;
    using fc = tiny_dnn::fully_connected_layer;
    using max_pool = tiny_dnn::max_pooling_layer;
    using batch_norm = tiny_dnn::batch_normalization_layer;
    using dropout = tiny_dnn::dropout_layer;
    using relu = tiny_dnn::relu_layer;
    using softmax = tiny_dnn::softmax_layer;

    using tiny_dnn::core::connection_table;
    using padding = tiny_dnn::padding;
    tiny_dnn::network<tiny_dnn::sequential> model;
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::backend_t::internal;

    const int n_fmaps1 = 32; // number of feature maps for upper layer
    const int n_fmaps2 = 48; // number of feature maps for lower layer
    const int n_fc = 512;  //number of hidden units in fully-connected layer

    const int input_w = 45;
    const int input_h = 45;
    const int input_c = 1;

    const int num_classes = 43;

    model << batch_norm(input_w * input_h, input_c)
          << conv(input_w, input_h, 3, 3, input_c, n_fmaps1, tiny_dnn::padding::same, true, 2, 2, 0, 0)  // 3x3 kernel, 2 stride

          << batch_norm(23 * 23, n_fmaps1)
          << relu()
          << conv(23, 23, 3, 3, n_fmaps1, n_fmaps1, tiny_dnn::padding::same)  // 3x3 kernel, 1 stride

          << batch_norm(23 * 23, n_fmaps1)
          << relu()
          << max_pool(23, 23, n_fmaps1, 2, 1, false)
          << conv(22, 22, 3, 3, n_fmaps1, n_fmaps2, tiny_dnn::padding::same, true, 2, 2)  // 3x3 kernel, 2 stride

          << batch_norm(11 * 11, n_fmaps2)
          << relu()
          << conv(11, 11, 3, 3, n_fmaps2, n_fmaps2, tiny_dnn::padding::same)  // 3x3 kernel, 1 stride

          << batch_norm(11 * 11, n_fmaps2)
          << relu()
          << max_pool(11, 11, n_fmaps2, 2, 1, false)
          << fc(10 * 10 * n_fmaps2, n_fc)

          << batch_norm(1 * 1, n_fc)
          << relu()
          << dropout(n_fc, 0.5)
          << fc(n_fc, num_classes)
          << softmax();

    // Initialize TorchClient
    // SimpleFlwrClient client(CLIENT_ID, model, local_training_data, local_validation_data, local_test_data);
    Tiny_Dnn_Client client(CLIENT_ID, model, train_labels, test_labels, train_images, test_images, test_labels, test_images);

    // Define a server address
    std::string server_add = SERVER_URL;

    // Start client
    start::start_client(server_add, &client);

    return 0;
}

