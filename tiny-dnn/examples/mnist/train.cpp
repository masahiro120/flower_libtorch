// /*
//     Copyright (c) 2013, Taiga Nomi and the respective contributors
//     All rights reserved.

//     Use of this source code is governed by a BSD-style license that can be found
//     in the LICENSE file.
// */
// #include <iostream>
// #include <vector>
// #include <list>
// #include <tuple>

// #include "../../../tiny-dnn/tiny_dnn/tiny_dnn.h"
// using namespace std;

// static void construct_net(tiny_dnn::network<tiny_dnn::sequential> &nn,
//                           tiny_dnn::core::backend_t backend_type) {
// // connection table [Y.Lecun, 1998 Table.1]
// #define O true
// #define X false
//   // clang-format off
// static const bool tbl[] = {
//     O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
//     O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
//     O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
//     X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
//     X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
//     X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
// };
// // clang-format on
// #undef O
// #undef X

//   // construct nets
//   //
//   // C : convolution
//   // S : sub-sampling
//   // F : fully connected
//   // clang-format off
//   using fc = tiny_dnn::layers::fc;
//   using conv = tiny_dnn::layers::conv;
//   using ave_pool = tiny_dnn::layers::ave_pool;
//   using tanh = tiny_dnn::activation::tanh;

//   using tiny_dnn::core::connection_table;
//   using padding = tiny_dnn::padding;

//   nn << conv(32, 32, 5, 1, 6,   // C1, 1@32x32-in, 6@28x28-out
//              padding::valid, true, 1, 1, 1, 1, backend_type)
//      << tanh()
//      << ave_pool(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
//      << tanh()
//      << conv(14, 14, 5, 6, 16,   // C3, 6@14x14-in, 16@10x10-out
//              connection_table(tbl, 6, 16),
//              padding::valid, true, 1, 1, 1, 1, backend_type)
//      << tanh()
//      << ave_pool(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
//      << tanh()
//      << conv(5, 5, 5, 16, 120,   // C5, 16@5x5-in, 120@1x1-out
//              padding::valid, true, 1, 1, 1, 1, backend_type)
//      << tanh()
//      << fc(120, 10, true, backend_type)  // F6, 120-in, 10-out
//      << tanh();
// }

// static void train_lenet(const std::string &data_dir_path,
//                         double learning_rate,
//                         const int n_train_epochs,
//                         const int n_minibatch,
//                         tiny_dnn::core::backend_t backend_type) {


//   // specify loss-function and learning strategy
//   tiny_dnn::network<tiny_dnn::sequential> nn;
//   tiny_dnn::adagrad optimizer;

//   construct_net(nn, backend_type);

//   std::cout << "load models..." << std::endl;

//   // load MNIST dataset
//   std::vector<tiny_dnn::label_t> train_labels, test_labels;
//   std::vector<tiny_dnn::vec_t> train_images, test_images;

//   tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
//                                &train_labels);
//   tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
//                                &train_images, -1.0, 1.0, 2, 2);
//   tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
//                                &test_labels);
//   tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
//                                &test_images, -1.0, 1.0, 2, 2);

//   std::cout << "start training" << std::endl;

//   tiny_dnn::progress_display disp(train_images.size());
//   tiny_dnn::timer t;

//   optimizer.alpha *=
//     std::min(tiny_dnn::float_t(4),
//              static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

//   int epoch = 1;
//   // create callback
//   auto on_enumerate_epoch = [&]() {
//     std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
//               << t.elapsed() << "s elapsed." << std::endl;
//     ++epoch;
//     tiny_dnn::result res = nn.test(test_images, test_labels);
//     std::cout << res.num_success << "/" << res.num_total << std::endl;

//     disp.restart(train_images.size());
//     t.restart();
//   };

//   auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

//   // training
//   nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, n_minibatch,
//                           n_train_epochs, on_enumerate_minibatch,
//                           on_enumerate_epoch);

//   std::cout << "end training." << std::endl;

//   // test and show results
//   nn.test(test_images, test_labels).print_detail(std::cout);

//   float_t train_num_success = 0;
//   float_t train_num_total = 0;

//   cout << "Train date size : " << train_images.size() << endl;
//   cout << "Train date size type : " << typeid(train_images.size()).name() << endl;
//   size_t train_size = train_images.size();
//   cout << "Train date size_t : " << train_size << endl;
//   cout << "Train date size_t type : " << typeid(train_size).name() << endl;

//   float_t train_accuracy = nn.calculate(train_images, train_labels, train_num_success, train_num_total);
//   cout << "Train  Accuracy : " << train_accuracy * 100 << "% (" << train_num_success << "/" << train_num_total << ")" << endl;
  
//   float_t test_num_success = 0;
//   float_t test_num_total = 0;
//   cout << "Test  date size : " << test_images.size() << endl;
//   cout << "Test date size type : " << typeid(test_images.size()).name() << endl;
//   size_t test_size = test_images.size();
//   cout << "Test date size_t : " << test_size << endl;
//   cout << "Test date size_t type : " << typeid(test_size).name() << endl;

//   float_t test_accuracy = nn.calculate(test_images, test_labels, test_num_success, test_num_total);
//   cout << "Test   Accuracy : " <<  test_accuracy * 100 << "% (" << test_num_success << "/" << test_num_total << ")" << endl;

//   cout << "train_images type : " << typeid(train_images).name() << endl;
//   cout << "train_labels type : " << typeid(train_labels).name() << endl;
//   cout << "test_images type : " << typeid(test_images).name() << endl;
//   cout << "test_labels type : " << typeid(test_labels).name() << endl;
  
//   // cout << "train_images.size() : " << train_images.size() << endl;
//   // cout << "train_labels.size() : " << train_labels.size() << endl;
//   // for(int i = 0;i < train_labels.size();i++){
//   //   cout << "train_labels : " << train_labels[i] << endl;
//   // }
  

//   // train_mse = nn.get_loss<tiny_dnn::mse>(train_images, train_labels, train_loss_count);
//   // train_mse = nn.test(train_images, train_labels).print_detail(std::cout);

//   // cout << "train_loss_count : " << train_loss_count << endl;
//   // cout << "train_mse : " << train_mse << endl;

//   // test_mse = nn.get_loss<tiny_dnn::mse>(test_images, test_labels, test_loss_count);

//   // cout << "test_loss_count : " << test_loss_count << endl;
//   // cout << "test_mse : " << test_mse << endl;

  

//   // save network model & trained weights
//   nn.save("model.bin");
// }

// static tiny_dnn::core::backend_t parse_backend_name(const std::string &name) {
//   const std::array<const std::string, 5> names = {{
//     "internal", "nnpack", "libdnn", "avx", "opencl",
//   }};
//   for (size_t i = 0; i < names.size(); ++i) {
//     if (name.compare(names[i]) == 0) {
//       return static_cast<tiny_dnn::core::backend_t>(i);
//     }
//   }
//   return tiny_dnn::core::default_engine();
// }

// static void usage(const char *argv0) {
//   std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
//             << " --learning_rate 1"
//             << " --epochs 30"
//             << " --minibatch_size 16"
//             << " --backend_type internal" << std::endl;
// }

// // int main(int argc, char **argv) {
// //   double learning_rate                   = 1;
// //   int epochs                             = 30;
// //   std::string data_path                  = "";
// //   int minibatch_size                     = 16;
// //   tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();


// //   if (argc == 2) {
// //     std::string argname(argv[1]);
// //     if (argname == "--help" || argname == "-h") {
// //       usage(argv[0]);
// //       return 0;
// //     }
// //   }
// //   for (int count = 1; count + 1 < argc; count += 2) {
// //     std::string argname(argv[count]);
// //     if (argname == "--learning_rate") {
// //       learning_rate = atof(argv[count + 1]);
// //     } else if (argname == "--epochs") {
// //       epochs = atoi(argv[count + 1]);
// //     } else if (argname == "--minibatch_size") {
// //       minibatch_size = atoi(argv[count + 1]);
// //     } else if (argname == "--backend_type") {
// //       backend_type = parse_backend_name(argv[count + 1]);
// //     } else if (argname == "--data_path") {
// //       data_path = std::string(argv[count + 1]);
// //     } else {
// //       std::cerr << "Invalid parameter specified - \"" << argname << "\""
// //                 << std::endl;
// //       usage(argv[0]);
// //       return -1;
// //     }
// //   }
// //   if (data_path == "") {
// //     std::cerr << "Data path not specified." << std::endl;
// //     usage(argv[0]);
// //     return -1;
// //   }
// //   if (learning_rate <= 0) {
// //     std::cerr
// //       << "Invalid learning rate. The learning rate must be greater than 0."
// //       << std::endl;
// //     return -1;
// //   }
// //   if (epochs <= 0) {
// //     std::cerr << "Invalid number of epochs. The number of epochs must be "
// //                  "greater than 0."
// //               << std::endl;
// //     return -1;
// //   }
// //   if (minibatch_size <= 0 || minibatch_size > 60000) {
// //     std::cerr
// //       << "Invalid minibatch size. The minibatch size must be greater than 0"
// //          " and less than dataset size (60000)."
// //       << std::endl;
// //     return -1;
// //   }
// //   std::cout << "Running with the following parameters:" << std::endl
// //             << "Data path: " << data_path << std::endl
// //             << "Learning rate: " << learning_rate << std::endl
// //             << "Minibatch size: " << minibatch_size << std::endl
// //             << "Number of epochs: " << epochs << std::endl
// //             << "Backend type: " << backend_type << std::endl
// //             << std::endl;
// //   try {
// //     train_lenet(data_path, learning_rate, epochs, minibatch_size, backend_type);
// //   } catch (tiny_dnn::nn_error &err) {
// //     std::cerr << "Exception: " << err.what() << std::endl;
// //   }


// //   // cout << "main train_loss_count : " << train_loss_count << endl;
// //   // cout << "main train_mse : " << train_mse << endl << endl;

// //   // cout << "main test_loss_count : " << test_loss_count << endl;
// //   // cout << "main test_mse : " << test_mse << endl << endl;


// //   // tiny_dnn::network<tiny_dnn::sequential> nn;
// //   // nn.load("LeNet-model");
// //   // vector<vector<vector<float, tiny_dnn::aligned_allocator<float, 64>>>> w_all;

// //   // for(int i = 0;i < 12;i++){
// //   //   cout << i << endl;
// //   //   vector<vector<float, tiny_dnn::aligned_allocator<float, 64>>> w;
// //   //   auto weights = nn[i]->weights();
// //   //   if(weights.size() != 0){
// //   //     w.push_back(*weights[0]);
// //   //     w.push_back(*weights[1]);
// //   //   }

// //   //   w_all.push_back(w);
// //   // }

// //   // cout << typeid(w_all).name() << endl;
// //   // cout << w_all.size() << endl;
// //   // cout << typeid(w_all[0]).name() << endl;
// //   // cout << w_all[0].size() << endl;
// //   // cout << typeid(w_all[0][0]).name() << endl;
// //   // cout << w_all[0][0].size() << endl;
// //   // cout << typeid(w_all[0][0][0]).name() << endl;
// //   // // cout << w_all[0][0][0].size() << endl;
  
// //   // cout << w_all[0][0][0] << endl;

// //   // // list<list<list<float>>> l(w_all.begin(), w_all.end());
  
// //   // cout << typeid(w_all).name() << endl;
// //   // cout << "w_all : " << w_all.size() << endl << endl;
// //   // for(int i = 0;i < w_all.size();i++){
// //   //   cout << "w_all[" << i << "] : " << w_all[i].size() << endl;
// //   //   if(w_all[i].size() != 0){
// //   //     for(int j = 0;j < w_all[i].size();j++){
// //   //       cout << "w_all[" << i << "][" << j << "] : " << w_all[i][j].size() << endl;

// //   //     }
// //   //   }
// //   //   cout << endl;
// //   // }

// //   // for(auto ele : w_all[0][0])
// //   //   cout << ele << " ";

// //   // cout << endl;
// //   // cout << endl;

// //   // cout << *w_all[0][0].begin() << endl;
// //   // cout << endl;
// //   // cout << endl;


// //   // // std::list<std::list<std::list<float>>> list3d;
// //   // // for (const auto& plane : w_all) {
// //   // //   std::list<std::list<float>> list2d;
// //   // //   for (const auto& row : plane) {
// //   // //     list2d.push_back(std::list<float>(row.begin(), row.end()));
// //   // //   }
// //   // //   list3d.push_back(list2d);
// //   // // }

// //   // std::list<std::list<std::list<std::string>>> list3d;
// //   // for (const auto& plane : w_all) {
// //   //   std::list<std::list<std::string>> list2d;
// //   //   for (const auto& row : plane) {
// //   //     std::list<std::string> list1d;
// //   //     for (const auto& element : row) {
// //   //       std::stringstream ss;
// //   //       ss << element;
// //   //       list1d.push_back(ss.str());
// //   //     }
// //   //     list2d.push_back(list1d);
// //   //   }
// //   //   list3d.push_back(list2d);
// //   // }

// //   // auto plane_iter = list3d.begin();
// //   // auto row_iter = plane_iter->begin();
// //   // auto element_iter = row_iter->begin();
// //   // std::cout << *element_iter << std::endl;
// //   // cout << typeid(*element_iter).name() << endl; 
// //   // cout << endl;

// //   // for (const auto& plane : list3d) {
// //   //   cout << "{";
// //   //   for (const auto& row : plane) {
// //   //     cout << "{";
// //   //     for (const auto& element : row) {
// //   //       std::cout << "{" << element << "} ";
// //   //     }
// //   //     std::cout << "}" <<  std::endl;
// //   //   }
// //   //   std::cout << "}" <<  std::endl;
// //   // }



// //   // std::vector<std::vector<std::vector<float>>> w_vec;
// //   // for (const auto& plane : list3d) {
// //   //     std::vector<std::vector<float>> vector2d;
// //   //     for (const auto& row : plane) {
// //   //       std::vector<float> vector1d;
// //   //       for (const auto& element : row) {
// //   //         std::stringstream ss(element);
// //   //         float value;
// //   //         ss >> value;
// //   //         vector1d.push_back(value);
// //   //       }
// //   //       vector2d.push_back(vector1d);
// //   //     }
// //   //     w_vec.push_back(vector2d);
// //   // }

// //   // cout << typeid(w_vec).name() << endl;
// //   // cout << w_vec.size() << endl;



// //   return 0;
// // }
