#pragma once
#include <stdint.h>
#include <math.h>

#define EMBEDDING_SIZE   32
#define MAX_CLASSES      10
#define MAX_ENROLLMENTS  5   // embeddings averaged per class

struct MatchResult {
    int   index;
    float distance;
};

class EmbeddingClassifier {
public:
    EmbeddingClassifier(float threshold);

    // Enroll a new embedding for a named class.
    // Call multiple times per class — they get averaged into one prototype.
    bool enroll(const char *label, const float *embedding);

    // Find nearest class. Returns index and distance.
    // index == -1 means no class was close enough (unknown).
    MatchResult match(const float *embedding);

    int   getClassCount()              const { return m_class_count; }
    const char *getLabel(int index)    const { return m_labels[index]; }

private:
    float l2distance(const float *a, const float *b);

    float  m_prototypes[MAX_CLASSES][EMBEDDING_SIZE];
    char   m_labels[MAX_CLASSES][32];
    int    m_enrollment_counts[MAX_CLASSES];
    int    m_class_count;
    float  m_threshold;
};