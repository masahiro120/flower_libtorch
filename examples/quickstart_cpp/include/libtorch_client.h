#pragma once
#include "client.h"
#include <ctime>
#include <memory>
#include <string>
#include <tuple>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>

#include "pointnet_model.h"

using namespace std;
/**
 * Validate the network on the entire test set
 *
 */

class Libtorch_Client : public flwr::Client {
 public:
  Libtorch_Client(std::string client_id,
                  Pointnet &model,
                  std::vector<int64_t> &train_label,
                  std::vector<int64_t> &test_label,
                  std::vector<float> &train_data,
                  std::vector<float> &test_data,
                  std::vector<int64_t> &default_test_label,
                  std::vector<float> &default_test_data);

  // void list3d_to_1d_str(std::list<std::list<std::list<string>>> mat, std::list<string> &ele, std::list<float> &num);
  // std::list<std::list<std::list<string>>> list1d_to_3d_str(std::list<string> ele, std::list<float> num);
  
  void set_parameters(flwr::Parameters params);

  virtual flwr::ParametersRes get_parameters() override;
  virtual flwr::PropertiesRes get_properties(flwr::PropertiesIns ins) override;
  virtual flwr::EvaluateRes evaluate(flwr::EvaluateIns ins) override;
  virtual flwr::FitRes fit(flwr::FitIns ins) override;


 private:
  int64_t client_id;
  Pointnet &model;
  std::vector<int64_t> &train_label;
  std::vector<int64_t> &test_label;
  std::vector<float> &train_data;
  std::vector<float> &test_data;
  std::vector<int64_t> &default_test_label;
  std::vector<float> &default_test_data;
  std::list<float> ele_num;

};
