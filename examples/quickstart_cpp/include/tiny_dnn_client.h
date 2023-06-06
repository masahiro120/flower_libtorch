/***********************************************************************************************************
 * 
 * @file libtorch_client.h
 *
 * @brief Define an example flower client, train and test method
 *
 * @version 1.0
 *
 * @date 06/09/2021
 *
 * ********************************************************************************************************/
#pragma once
#include "client.h"
#include "line_fit_model.h"
#include "../../../tiny-dnn/tiny_dnn/tiny_dnn.h"
#include <ctime>
#include <memory>
#include <string>
#include <tuple>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>

using namespace std;
/**
 * Validate the network on the entire test set
 *
 */

class Tiny_Dnn_Client : public flwr::Client {
 public:
  Tiny_Dnn_Client(std::string client_id,
                   tiny_dnn::network<tiny_dnn::sequential> &model,
                   std::vector<tiny_dnn::label_t> &train_labels,
                   std::vector<tiny_dnn::label_t> &test_labels,
                   std::vector<tiny_dnn::vec_t> &train_images,
                   std::vector<tiny_dnn::vec_t> &test_images,
                   std::vector<tiny_dnn::label_t> &default_test_labels,
                   std::vector<tiny_dnn::vec_t> &default_test_images);

  // void list3d_to_1d_str(std::list<std::list<std::list<string>>> mat, std::list<string> &ele, std::list<float> &num);
  // std::list<std::list<std::list<string>>> list1d_to_3d_str(std::list<string> ele, std::list<float> num);
  
  void set_parameters(flwr::Parameters params);

  virtual flwr::ParametersRes get_parameters() override;
  virtual flwr::PropertiesRes get_properties(flwr::PropertiesIns ins) override;
  virtual flwr::EvaluateRes evaluate(flwr::EvaluateIns ins) override;
  virtual flwr::FitRes fit(flwr::FitIns ins) override;


 private:
  int64_t client_id;
  tiny_dnn::network<tiny_dnn::sequential> &model;
  std::vector<tiny_dnn::label_t> &train_labels;
  std::vector<tiny_dnn::label_t> &test_labels;
  std::vector<tiny_dnn::vec_t> &train_images;
  std::vector<tiny_dnn::vec_t> &test_images;
  std::vector<tiny_dnn::label_t> &default_test_labels;
  std::vector<tiny_dnn::vec_t> &default_test_images;
  std::list<float> ele_num;

};
