#include <algorithm>
#include <random>

// #include "tiny_dnn_client.h"
#include "start.h"
// #include "../../../tiny-dnn/tiny_dnn/tiny_dnn.h"
// #include "../../../tiny-dnn/tiny_dnn/io/layer_factory.h"

#include "pointnet_model.h"
#include "libtorch_client.h"

// #include "half_function.cc"

extern float train_dataset[1622][2048][3];
extern int train_labelset[1622];
extern float test_dataset[405][2048][3];
extern int test_labelset[405];

int DATA_SIZE = 100;

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

    // // シャッフル
    // std::vector<tiny_dnn::vec_t> train_images(data_per_class * classes / 2 * 0.8);
    // std::vector<tiny_dnn::vec_t> test_images(data_per_class * classes / 2 * 0.2);
    // std::vector<tiny_dnn::label_t> train_labels(data_per_class * classes / 2 * 0.8);
    // std::vector<tiny_dnn::label_t> test_labels(data_per_class * classes / 2 * 0.2);
    // // std::random_device seed_gen;
    // // std::mt19937 engine(seed_gen());

    // const unsigned int seed = 123;
    // std::mt19937 engine(seed);

    // std::vector<int> num_list;

    // for (int i=0; i< data_per_class * classes; i++) num_list.push_back(i);
    // std::shuffle(num_list.begin(), num_list.end(), engine);

    // // if(CLIENT_ID == "0") {
    // //   printf("Client 0\n");
    // //   for(int i = 0;i < train_images_data.size() / 2;i++){
    // //     if (i % 5 == 0){
    // //       // test
    // //       test_images.push_back(train_images_data[num_list[i]]);
    // //       test_labels.push_back(train_labels_data[num_list[i]]);
    // //     } else {
    // //       // train
    // //       train_images.push_back(train_images_data[num_list[i]]);
    // //       train_labels.push_back(train_labels_data[num_list[i]]);
    // //     }
    // //   }
    // // } else {
    // //   printf("Client 1\n");
    // //   for(int i = train_images_data.size() / 2;i < train_images_data.size();i++){
    // //     if (i % 5 == 0){
    // //       // test
    // //       test_images.push_back(train_images_data[num_list[i]]);
    // //       test_labels.push_back(train_labels_data[num_list[i]]);
    // //     } else {
    // //       // train
    // //       train_images.push_back(train_images_data[num_list[i]]);
    // //       train_labels.push_back(train_labels_data[num_list[i]]);
    // //     }
    // //   }
    // // }
    // int test_index = 0;
    // int train_index = 0;

    // if(CLIENT_ID == "0") {
    //   printf("Client 0\n");
    //   for(int i = 0; i < data_per_class * classes / 2; i++){
    //     // printf("i: %d\n", i);
    //     // printf("num_list[i]: %d\n", num_list[i]);
    //     if (i % 5 == 0){
    //       // test
    //       test_images[test_index].resize(45 * 45);
    //       for(int j = 0; j < 45 * 45; j++){
    //         test_images[test_index][j] = train_images_data[num_list[i]][j];
    //       }
    //       test_labels[test_index] = train_labels_data[num_list[i]];
    //       test_index++;
    //     } else {
    //       // train
    //       train_images[train_index].resize(45 * 45);
    //       for(int j = 0; j < 45 * 45; j++){
    //         train_images[train_index][j] = train_images_data[num_list[i]][j];
    //       }
    //       train_labels[train_index] = train_labels_data[num_list[i]];
    //       train_index++;
    //     }
    //   }
    // } else {
    //   printf("Client 1\n");
    //   for(int i = data_per_class * classes / 2; i < data_per_class * classes; i++){
    //     if (i % 5 == 0){
    //       // test
    //       test_images[test_index].resize(45 * 45);
    //       for(int j = 0; j < 45 * 45; j++){
    //         test_images[test_index][j] = train_images_data[num_list[i]][j];
    //       }
    //       test_labels[test_index] = train_labels_data[num_list[i]];
    //       test_index++;
    //     } else {
    //       // train
    //       train_images[train_index].resize(45 * 45);
    //       for(int j = 0; j < 45 * 45; j++){
    //         train_images[train_index][j] = train_images_data[num_list[i]][j];
    //       }
    //       train_labels[train_index] = train_labels_data[num_list[i]];
    //       train_index++;
    //     }
    //   }
    // }

    // std::cout << "Train data images size : " << train_images.size() << std::endl;
    // std::cout << "Train data labels size : " << train_labels.size() << std::endl;

    // Define a model
    // pointnet
    Pointnet model(40);

    // data
    std::vector<int64_t> train_label;
    std::vector<int64_t> test_label;
    // std::vector<std::vector<std::vector<float>>> train_data;
    // std::vector<std::vector<std::vector<float>>> test_data;
    std::vector<float> train_data;
    std::vector<float> test_data;

    int train_data_num = DATA_SIZE * 0.8;
    int test_data_num = DATA_SIZE * 0.2;

    int POINT_NUM = 2048;
    int DIM = 3;

    // train_data.resize(train_data_num);
    // train_label.resize(train_data_num);
    // for(int i = 0; i < train_data_num; i++){
    //     train_data[i].resize(2048);
    //     for(int j = 0; j < 2048; j++){
    //         train_data[i][j].resize(3);
    //         for(int k = 0; k < 3; k++){
    //             train_data[i][j][k] = train_dataset[i][j][k];
    //         }
    //     }
    //     train_label[i] = train_labelset[i];
    // }

    // test_data.resize(test_data_num);
    // test_label.resize(test_data_num);
    // for(int i = 0; i < test_data_num; i++){
    //     test_data[i].resize(2048);
    //     for(int j = 0; j < 2048; j++){
    //         test_data[i][j].resize(3);
    //         for(int k = 0; k < 3; k++){
    //             test_data[i][j][k] = test_dataset[i][j][k];
    //         }
    //     }
    //     test_label[i] = test_labelset[i];
    // }

    for(int i = 0; i < train_data_num; i++){
        for(int j = 0; j < POINT_NUM; j++){
            for(int k = 0; k < DIM; k++){
                train_data.push_back(train_dataset[i][j][k]);
            }
        }
        train_label.push_back(train_labelset[i]);
    }

    for(int i = 0; i < test_data_num; i++){
        for(int j = 0; j < POINT_NUM; j++){
            for(int k = 0; k < DIM; k++){
                test_data.push_back(test_dataset[i][j][k]);
            }
        }
        test_label.push_back(test_labelset[i]);
    }

    // Initialize TorchClient
    Libtorch_Client client(CLIENT_ID, model, train_label, test_label, train_data, test_data, test_label, test_data);

    // Define a server address
    std::string server_add = SERVER_URL;

    // Start client
    start::start_client(server_add, &client);

    return 0;
}

