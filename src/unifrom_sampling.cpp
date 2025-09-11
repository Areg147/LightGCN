/*
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <random>
#include <algorithm>
#include <ctime>
#include <cstring>

namespace py = pybind11;

int randint_(int end) {
    return rand() % end;
}

py::array_t<int> uniform_sampler(int data_size,
                                 int users_count,
                                 std::map<int, std::vector<int>> user_pos_items_dict_train,
                                 int items_count,
                                 int neg_per_pos)
{
    py::array_t<int> result_array = py::array_t<int>({data_size, 2 + neg_per_pos});
    py::buffer_info buf = result_array.request();
    int* ptr = (int*)buf.ptr;

    int idx = 0;

    for (int i = 0; i < data_size * 2; ++i) {  
        int user = randint_(users_count);

        if (user_pos_items_dict_train.find(user) == user_pos_items_dict_train.end())
            continue;

        const std::vector<int>& pos_items = user_pos_items_dict_train[user];
        if (pos_items.empty())
            continue;

        int positem = pos_items[randint_(pos_items.size())];

        std::vector<int> negitems;
        while ((int)negitems.size() < neg_per_pos) {
            int negitem = randint_(items_count);
            if (std::find(pos_items.begin(), pos_items.end(), negitem) == pos_items.end() &&
                std::find(negitems.begin(), negitems.end(), negitem) == negitems.end()) {
                negitems.push_back(negitem);
            }
        }

        ptr[idx * (2 + neg_per_pos) + 0] = user;
        ptr[idx * (2 + neg_per_pos) + 1] = positem;
        for (int j = 0; j < neg_per_pos; ++j) {
            ptr[idx * (2 + neg_per_pos) + 2 + j] = negitems[j];
        }

        idx++;
        if (idx == data_size) break;
    }

    if (idx < data_size) {
        py::array_t<int> final_array = py::array_t<int>({idx, 2 + neg_per_pos});
        py::buffer_info final_buf = final_array.request();
        int* final_ptr = (int*)final_buf.ptr;
        std::memcpy(final_ptr, ptr, sizeof(int) * idx * (2 + neg_per_pos));
        return final_array;
    }

    return result_array;
}

void set_seed(unsigned int seed) {
    srand(seed);
}

PYBIND11_MODULE(unifrom_sampling, m) {
    srand(static_cast<unsigned>(time(0)));
    m.doc() = "Uniform sampler module";
    m.def("uniform_sampler", &uniform_sampler, "uniform sampler with N negatives",
          py::arg("data_size"),
          py::arg("users_count"),
          py::arg("user_pos_items_dict_train"),
          py::arg("items_count"),
          py::arg("neg_per_pos"));
    m.def("seed", &set_seed, "Set random seed", py::arg("seed"));
}