#include "activations.h"

ACTIVATION get_activation(std::string s) {
    if(s == "lhtan")        return LHTAN;
    if(s == "hardtan")      return HARDTAN;
    if(s == "hardtan")      return HARDTAN;
    if(s == "linear")       return LINEAR;
    if(s == "logistic")     return LOGISTIC;
    if(s == "relu")         return RELU;
    if(s == "elu")          return ELU;
    if(s == "selu")         return SELU;
    if(s == "leaky")        return LEAKY;
    if(s == "tanh")         return TANH;
    printf("Couldn't find activation function %s, going with ReLu.\n", s);
    return RELU;
}

float activate(float x, std::string s) 
{
    ACTIVATION a = get_activation(s);
    switch (a)
    {
    case LHTAN:         return lhtan_activate(x);
    case HARDTAN:       return hardtan_activate(x);
    case LINEAR:        return linear_activate(x);
    case LOGISTIC:      return logistic_activate(x);
    case RELU:          return relu_activate(x);
    case ELU:           return elu_activate(x);
    case SELU:          return selu_activate(x);
    case LEAKY:         return leaky_activate(x);
    case TANH:          return tanh_activate(x);
    }
}