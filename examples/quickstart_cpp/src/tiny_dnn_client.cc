#include "tiny_dnn_client.h"
#include "../../../tiny-dnn/tiny_dnn/tiny_dnn.h"
#include "../../../tiny-dnn/examples/mnist/train.cpp"
#include "../../../tiny-dnn/tiny_dnn/layers/layer.h"
#include "../../../tiny-dnn/tiny_dnn/util/weight_init.h"


using namespace std;
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;
/**
 * Initializer
 */
Tiny_Dnn_Client::Tiny_Dnn_Client(std::string client_id,
                                   tiny_dnn::network<tiny_dnn::sequential> &model,
                                   std::vector<tiny_dnn::label_t> &train_labels,
                                   std::vector<tiny_dnn::label_t> &test_labels,
                                   std::vector<tiny_dnn::vec_t> &train_images,
                                   std::vector<tiny_dnn::vec_t> &test_images,
                                   std::vector<tiny_dnn::label_t> &default_test_labels,
                                   std::vector<tiny_dnn::vec_t> &default_test_images)
        : model(model),
          train_labels(train_labels),
          train_images(train_images),
          test_labels(test_labels),
          test_images(test_images),
          default_test_labels(default_test_labels),
          default_test_images(default_test_images) {

};

/**
 * Return the current local model parameters
 * Simple string are used for now to test communication, needs updates in the future
 */
flwr::ParametersRes Tiny_Dnn_Client::get_parameters() {

    std::cout <<"get_parameters Test" << std::endl;
    // // Serialize

    vector<vector<vector<double>>> w_all;
    // vector<vector<vec_t>> w_all;

    for(int i = 0;i < this->model.depth();i++){
      if(i == 1 || i == 4 || i == 8 || i == 11 || i == 15|| i == 19){
        vector<vector<double>> w;
        auto weights = this->model[i]->weights();
        if(weights.size() != 0){
          vector<double> weights_double;
          vector<double> bias_double;
          for(int j = 0;j < weights[0]->size();j++){

            double ele = weights[0][0][j];
            weights_double.push_back(ele);
          }
          for(int j = 0;j < weights[1]->size();j++){
            double ele = weights[1][0][i];
            bias_double.push_back(ele);
          }
          // std::transform(*weights[0].begin(), *weights[0].end(), std::back_inserter(weights_double), [](float x) { return static_cast<double>(x); });
          // cout << "weights_double[0] : " << weights_double[0]<< endl;
          // cout << "bias_double[0] : " << bias_double[0]<< endl;
          w.push_back(weights_double);
          w.push_back(bias_double);
        } 
        // std::transform(*weights.begin(), *weights.end(), std::back_inserter(weights_double), [](float x) { return static_cast<double>(x); });
        w_all.push_back(w);
      }
    }
    // cout << "w_all[0][0][0] : " << w_all[0][0][0] << endl;
    // cout << "w_all[0][1][0] : " << w_all[0][1][0] << endl;
    // w_all = 
    // [
    //   [ [0.1,0.1,0.1...]<-w[0] [0.2,0.2,...]<-w[0] ] <-w
    //   [ [] [] ]
    //   .
    //   .
    //   .
    //   [ [] [] ]
    // ]

    std::list<std::string> tensors;

    for (int i = 0; i < w_all.size(); i++) {
      if(w_all[i].size() != 0){

        // std::copy(w_all[i][0].begin(), w_all[i][0].end(), std::front_inserter(lst));
        std::ostringstream oss1;
        oss1.write(reinterpret_cast<const char *>(w_all[i][0].data()), w_all[i][0].size() * sizeof(double));
        // if (i < 2){
        //   cout << "oss1 : " << oss1.str() << endl;
        // }
        tensors.push_back(oss1.str());

        std::ostringstream oss2;
        oss2.write(reinterpret_cast<const char *>(w_all[i][1].data()), w_all[i][1].size() * sizeof(double));

        // if (i < 2){
        //   cout << "oss2 : " << oss2.str() << endl;
        // }
        tensors.push_back(oss2.str());
        // printf("w_all[i][0].size() : %d, w_all[i][1].size() : %d\n", w_all[i][0].size(), w_all[i][1].size());
      }
    }

    std::string tensor_str = "cpp_double";

    std::cout <<"get_parameters Test end" << std::endl;

    return flwr::Parameters(tensors, tensor_str);
};

void Tiny_Dnn_Client::set_parameters(flwr::Parameters params) {

    std::cout <<"set_parameters Test" << std::endl;

    // for (size_t i = 0; i < this->model.depth(); i++) {
    //   cout << "Layer " << i << endl;
    //   cout << this->model[i]->in_size() << endl;
    //   cout << this->model[i]->out_size() << endl;
    // }

    std::list<std::string> s = params.getTensors();
    std::cout << "Received " << s.size() <<" Layers from server:" << std::endl;
    // std::list<float> ele_num = this->ele_num;
    // for (const auto& tensor : s) {
    //   std::cout << tensor.size() << std::endl;
    //   std::cout << tensor[0] << std::endl;
    // }

    // [0.1,0.1,0.1] -> [b"sasa"]

    int w_size;
    int b_size;
    int x;

    // s = [ [b"sasa"]  [b] [] ...]
    // Check if the list is not empty
    if (s.empty() == 0) {
        // Loop through the list of tensors
        for (int i = 0; i < s.size(); i += 2) {

            if(i == 0){
              w_size = 288;
              b_size = 32;
            } else if(i == 2) {
              w_size = 9216;
              b_size = 32;
            } else if(i == 4) {
              w_size = 13824;
              b_size = 48;
            } else if(i == 6) {
              w_size = 20736;
              b_size = 48;
            } else if(i == 8) {
              w_size = 2457600;
              b_size = 512;
            } else if(i == 10) {
              w_size = 22016;
              b_size = 43;
            }
            auto layer = std::next(s.begin(), i);
            size_t num_bytes = (*layer).size();
            const char *weights_char = (*layer).c_str();
            const double *weights_double = reinterpret_cast<const double *>(weights_char);

            vec_t weights;

            for(int j = 0;j < w_size;j++){
              weights.push_back(*(weights_double + j));
            }

            auto layer_2 = std::next(s.begin(), i + 1);
            num_bytes = (*layer_2).size();
            const char *bias_char = (*layer_2).c_str();
            const double *bias_double = reinterpret_cast<const double *>(bias_char);

            vec_t bias;

            for(int j = 0;j < b_size;j++){
              bias.push_back(*(bias_double + j));
            }

            if(i == 0){
              x = 1;
            } else if(i == 2) {
              x = 4;
            } else if(i == 4) {
              x = 8;
            } else if(i == 6) {
              x = 11;
            } else if(i == 8) {
              x = 15;
            } else if(i == 10) {
              x = 19;
            }

            if(i == 0) {
              auto before = this->model[1]->weights();
              cout << "before weights : " << before[0][0][0] << endl;
              cout << "before   bias  : " << before[1][0][0] << endl;
            }

            this->model[x]->weight_bias_set(weights, bias);

            if(i == 0){
              auto after = this->model[1]->weights();
              cout << "after  weights : " << after[0][0][0] << endl;
              cout << "after    bias  : " << after[1][0][0] << endl;
            }

            // after = [
                    //   [ [ 0.1,0.1,.... ] ] 重み
                    //   [ [0.2,0.2,0.2,0.2,0.2,0.2...] ]バイアス
                    // ]
            w_size = 0;
            b_size = 0;
        }
    }

    std::cout <<"set_parameters Test end" << std::endl;
};

flwr::PropertiesRes Tiny_Dnn_Client::get_properties(flwr::PropertiesIns ins) {
    
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
flwr::FitRes Tiny_Dnn_Client::fit(flwr::FitIns ins) {
    std::cout << "Fitting..." << std::endl;
    flwr::FitRes resp;

    flwr::Parameters p = ins.getParameters();
    std::list<std::string> p_tensor = p.getTensors();
    this->set_parameters(p);

    double learning_rate                   = 0.01;
    int n_train_epochs                     = 1;
    int n_minibatch                        = 32;
    tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();


    // float_t train_num_success = 0;
    // float_t train_num_total = 0;

    tiny_dnn::adagrad optimizer;

    // construct_net(this->model, backend_type);

    std::cout << "start training" << std::endl;

    tiny_dnn::progress_display disp(train_images.size());
    tiny_dnn::timer t;
    optimizer.alpha *=
      std::min(tiny_dnn::float_t(4),
               static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate));

    int epoch = 1;
    // create callback

    auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };
    auto on_enumerate_epoch = [&]() {
      std::cout << " Epoch " << epoch << "/" << n_train_epochs << " finished. " << t.elapsed() << "s elapsed." << std::endl;

      // lossの計算
      // std::cout << "calculate loss..." << std::endl;
      // auto train_loss = this->model.get_loss<tiny_dnn::mse>(train_images, train_labels);
      // auto test_loss = this->model.get_loss<tiny_dnn::mse>(test_images, test_labels);

      // // accuracyの計算
      // std::cout << "calculate accuracy..." << std::endl;
      // tiny_dnn::result train_results = this->model.test(train_images, train_labels);
      // tiny_dnn::result test_results = this->model.test(test_images, test_labels);
      // float_t train_accuracy = (float_t)train_results.num_success * 100 / train_results.num_total;
      // float_t test_accuracy = (float_t)test_results.num_success * 100 / test_results.num_total;

      // std::cout << "train loss: " << train_loss << " test loss: " << test_loss << std::endl;
      // std::cout << "train accuracy: " << train_accuracy << "% test accuracy: " << test_accuracy << "%" << std::endl;

      ++epoch;
      disp.restart(train_images.size());
      t.restart();
    };

    // std::tuple<size_t, float, double> result = this->model.train_SGD(this->training_dataset);
    this->model.train<tiny_dnn::cross_entropy_multiclass>(optimizer, train_images, train_labels, n_minibatch,
                            n_train_epochs, on_enumerate_minibatch,
                            on_enumerate_epoch);

    // todo: predict()

    // float_t train_accuracy = this->model.calc_cross_entoropy(this->train_images, this->train_labels);

    // cout << "Train_accuracy : " << train_accuracy << endl;
    // float_t train_accuracy = this->model.calculate(this->test_images, this->test_labels, train_num_success, train_num_total);
    // std::cout << "Train  Accuracy : " << train_accuracy * 100 << "% (" << train_num_success << "/" << train_num_total << ")" << std::endl;
    size_t train_size = this->train_images.size();

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
flwr::EvaluateRes Tiny_Dnn_Client::evaluate(flwr::EvaluateIns ins) {
    std::cout << "evaluate Test" << std::endl;

    std::cout << "Evaluating..." << std::endl;
    flwr::EvaluateRes resp;
    flwr::Parameters p = ins.getParameters();
    this->set_parameters(p);

    float_t test_num_success = 0;
    float_t test_num_total = 0;


    // std::cout << "calculate accuracy..." << std::endl;
    // tiny_dnn::result test_results = this->model.test(default_test_images, default_test_labels);
    // float_t test_accuracy = (float_t)test_results.num_success * 100 / test_results.num_total;
    // std::cout << "test accuracy: " << test_accuracy << "%" << std::endl;
    std::cout << "calculate loss..." << std::endl;
    float_t test_loss = this->model.get_loss<tiny_dnn::mse>(default_test_images, default_test_labels);
    std::cout << "loss = " << test_loss << std::endl; 

    // lossの計算
    // std::cout << "calculate loss..." << std::endl;
    // auto train_loss = this->model.get_loss<tiny_dnn::mse>(train_images, train_labels);
    // auto test_loss = this->model.get_loss<tiny_dnn::mse>(test_images, test_labels);

    // // accuracyの計算
    // std::cout << "calculate accuracy..." << std::endl;
    // tiny_dnn::result train_results = this->model.test(train_images, train_labels);
    // tiny_dnn::result test_results = this->model.test(test_images, test_labels);
    // float_t train_accuracy = (float_t)train_results.num_success * 100 / train_results.num_total;
    // float_t test_accuracy = (float_t)test_results.num_success * 100 / test_results.num_total;

    // std::cout << "train loss: " << train_loss << " test loss: " << test_loss << std::endl;
    // std::cout << "train accuracy: " << train_accuracy << "% test accuracy: " << test_accuracy << "%" << std::endl;


    // // float_t test_accuracy = this->model.calculate(this->default_test_images, this->default_test_labels, test_num_success, test_num_total);
    // std::cout << "Test  Accuracy : " << test_accuracy * 100 << "% (" << test_num_success << "/" << test_num_total << ")" << std::endl;
    // float_t test_loss = this->model.calc_cross_entoropy(this->default_test_images, this->default_test_labels);
    size_t test_size = default_test_images.size();

    std::tuple<size_t, float, float> result = std::make_tuple(test_size, test_loss, test_loss);

    resp.setNum_example(std::get<0>(result));
    resp.setLoss(std::get<1>(result));

    flwr::Scalar loss_metric = flwr::Scalar();
    loss_metric.setFloat(std::get<2>(result));
    std::map<std::string, flwr::Scalar> metric = {{"loss", loss_metric}};
    resp.setMetrics(metric);

    std::cout <<"evaluate Test end" << std::endl;

    return resp;

};


