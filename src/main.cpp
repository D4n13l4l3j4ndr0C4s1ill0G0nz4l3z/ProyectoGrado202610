#include "I2SMicSampler.h"
#include "AudioProcessor.h"
#include "NeuralNetwork.h"
#include "NimBLEDevice.h"

#include <Arduino.h>
#include <math.h>

#define WINDOW_SIZE      320
#define STEP_SIZE        160
#define POOLING_SIZE     6
#define AUDIO_LENGTH     16000
#define SPECTROGRAM_ROWS 98
#define SPECTROGRAM_COLS 43
#define SPECTROGRAM_SIZE (SPECTROGRAM_ROWS * SPECTROGRAM_COLS)
#define EMBEDDING_SIZE   32

#define NUM_CLASSES      4
#define INFER_BUTTON_PIN 4

#define TOUCH_THRESHOLD 40

#define SERVICE_UUID        "12345678-1234-1234-1234-123456789abc"
#define CHARACTERISTIC_UUID "abcdefab-1234-5678-1234-abcdefabcdef"

NimBLECharacteristic* pCharacteristic = nullptr;


static bool isTouched(int pin) {
    return touchRead(pin) < TOUCH_THRESHOLD;
}

static const int CLASS_BUTTON_PINS[NUM_CLASSES] = {13, 12, 14, 15};
static const String CONTROLMESSAGES[NUM_CLASSES] = {"AD100","AT100","GH100","GA100"};

i2s_pin_config_t i2sPins = {
    .bck_io_num   = 26,
    .ws_io_num    = 27,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = 22
};

I2SMicSampler  i2sSampler(i2sPins, false);
AudioProcessor audioProcessor(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
NeuralNetwork *neuralNetwork = nullptr;
float         *spectrogram   = nullptr;
TaskHandle_t   processorTaskHandle;

static volatile bool spectrogramReady = false;


static float classEmbedding[NUM_CLASSES][EMBEDDING_SIZE]; // running mean embedding
static int   classSampleCount[NUM_CLASSES];               // samples accumulated per class


static void l2normalize(float *vec, int len)
{
    float norm = 0.0f;
    for (int i = 0; i < len; i++)
        norm += vec[i] * vec[i];
    norm = sqrtf(norm);
    if (norm < 1e-8f) return;
    for (int i = 0; i < len; i++)
        vec[i] /= norm;
}

static float euclideanDistance(const float *a, const float *b, int len)
{
    float dist = 0.0f;
    for (int i = 0; i < len; i++)
    {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return sqrtf(dist);
}


static bool computeNormalizedEmbedding(float *out)
{
    float *modelInput = neuralNetwork->getInputBuffer();
    if (!modelInput)
    {
        Serial.println("ERROR: model input buffer is null");
        return false;
    }
    memcpy(modelInput, spectrogram, SPECTROGRAM_SIZE * sizeof(float));
    float *embedding = neuralNetwork->getEmbedding();
    memcpy(out, embedding, EMBEDDING_SIZE * sizeof(float));
    l2normalize(out, EMBEDDING_SIZE);
    return true;
}


static void updateClassEmbedding(int classIdx, const float *newEmbedding)
{
    classSampleCount[classIdx]++;
    int n = classSampleCount[classIdx];
    float *mean = classEmbedding[classIdx];

    for (int i = 0; i < EMBEDDING_SIZE; i++)
        mean[i] += (newEmbedding[i] - mean[i]) / (float)n;

    l2normalize(mean, EMBEDDING_SIZE);
}


void processorTask(void *param)
{
    I2SMicSampler *sampler = (I2SMicSampler *)param;

    int buffersToWait = AUDIO_LENGTH / SAMPLE_BUFFER_SIZE;
    for (int i = 0; i < buffersToWait; i++)
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    while (true)
    {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        RingBufferAccessor *reader = sampler->getRingBufferReader();
        reader->rewind(AUDIO_LENGTH);
        audioProcessor.get_spectrogram(reader, spectrogram);
        delete reader;

        spectrogramReady = true;
    }
}

void buttonTask(void *param)
{
    bool lastClassState[NUM_CLASSES];
    bool lastInferState = HIGH;

    for (int i = 0; i < NUM_CLASSES; i++)
        lastClassState[i] = HIGH;

    while (true)
    {
        for (int c = 0; c < NUM_CLASSES; c++)
        {
            bool current = isTouched(CLASS_BUTTON_PINS[c]) ? LOW : HIGH;

            if (lastClassState[c] == HIGH && current == LOW) // falling edge
            {
                if (!spectrogramReady)
                {
                    Serial.printf("Class %d button: not ready yet.\n", c);
                }
                else
                {
                    float embedding[EMBEDDING_SIZE];
                    if (computeNormalizedEmbedding(embedding))
                    {
                        bool isFirst = (classSampleCount[c] == 0);
                        updateClassEmbedding(c, embedding);

                        if (isFirst)
                            Serial.printf("Class %d: first embedding stored.\n", c);
                        else
                            Serial.printf("Class %d: embedding updated (n=%d).\n",
                                          c, classSampleCount[c]);

                        // Send embedding over serial
                        uint8_t embedmarker[] = {0x11, 0x22, 0x33, 0x44};
                        Serial.write(embedmarker, 4);
                        Serial.write((uint8_t *)&c, 1);  // class index byte
                        Serial.write((uint8_t *)embedding, EMBEDDING_SIZE * sizeof(float));
                    }
                }
            }

            lastClassState[c] = current;
        }

        bool currentInfer = isTouched(INFER_BUTTON_PIN) ? LOW : HIGH;

        if (lastInferState == HIGH && currentInfer == LOW) // falling edge
        {
            if (!spectrogramReady)
            {
                Serial.println("Inference button: not ready yet.");
            }
            else
            {
                float embedding[EMBEDDING_SIZE];
                if (computeNormalizedEmbedding(embedding))
                {
                    // Find nearest enrolled class
                    int   bestClass = -1;
                    float bestDist  = 1e9f;

                    for (int c = 0; c < NUM_CLASSES; c++)
                    {
                        if (classSampleCount[c] == 0) continue; // not yet enrolled

                        float dist = euclideanDistance(embedding, classEmbedding[c], EMBEDDING_SIZE);
                        Serial.printf("  Distance to class %d: %.6f\n", c, dist);

                        if (dist < bestDist)
                        {
                            bestDist  = dist;
                            bestClass = c;
                        }
                    }

                    if (bestClass < 0)
                        Serial.println("No classes enrolled yet.");
                    else{
                        Serial.printf(">>> Best match: Class %d (distance %.6f)\n", bestClass, bestDist);
                        pCharacteristic->setValue(CONTROLMESSAGES[bestClass]);
                        pCharacteristic->notify();
                    }
                }
            }
        }

        lastInferState = currentInfer;
        vTaskDelay(20 / portTICK_PERIOD_MS);
    }
}

void setup()
{
    Serial.begin(921600);
    delay(500);
    NimBLEDevice::init("CONTROLLER");
    
    NimBLEServer *pServer = NimBLEDevice::createServer();
    NimBLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(CHARACTERISTIC_UUID);

    pService->start();
    pCharacteristic->setValue("ESPERANDO COMANDO DE VOZ...");
    
    NimBLEAdvertising *pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID); // advertise the UUID of our service
    pAdvertising->setName("CONTROLLERESP32"); // advertise the device name
    pAdvertising->start(); 

    memset(classEmbedding,   0, sizeof(classEmbedding));
    memset(classSampleCount, 0, sizeof(classSampleCount));

    spectrogram = (float *)malloc(SPECTROGRAM_SIZE * sizeof(float));
    if (!spectrogram) {
        Serial.println("ERROR: spectrogram malloc failed");
        while (true) {}
    }

    neuralNetwork = new NeuralNetwork();

    xTaskCreate(processorTask, "Processor", 16384, &i2sSampler, 1, &processorTaskHandle);
    xTaskCreate(buttonTask,    "Button",     4096,  nullptr,     1, nullptr);

    i2s_config_t i2sConfig = {
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate          = 16000,
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count        = 4,
        .dma_buf_len          = 1024,
        .use_apll             = false,
        .tx_desc_auto_clear   = false,
        .fixed_mclk           = 0
    };
    ((I2SSampler &)i2sSampler).start(I2S_NUM_0, i2sConfig, processorTaskHandle);
}

void loop()
{
    vTaskDelay(1000 / portTICK_PERIOD_MS);
}
