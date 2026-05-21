#ifndef __NeuralNetwork__
#define __NeuralNetwork__

#include <stdint.h>

namespace tflite
{
    class AllOpsResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} // namespace tflite

struct TfLiteTensor;

typedef struct
{
    float score;
    int index;
} NNResult;

class NeuralNetwork
{
private:
    tflite::AllOpsResolver *m_resolver;
    tflite::ErrorReporter *m_error_reporter;
    const tflite::Model *m_model;
    tflite::MicroInterpreter *m_interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    uint8_t *m_tensor_arena;

public:
    NeuralNetwork();
    ~NeuralNetwork();
    float *getInputBuffer();
    float *getOutputBuffer();
    float *getEmbedding();
    NNResult predict();
};

#endif