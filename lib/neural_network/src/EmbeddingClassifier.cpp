#include "EmbeddingClassifier.h"
#include <string.h>
#include <float.h>

EmbeddingClassifier::EmbeddingClassifier(float threshold)
    : m_class_count(0), m_threshold(threshold)
{
    memset(m_prototypes,        0, sizeof(m_prototypes));
    memset(m_enrollment_counts, 0, sizeof(m_enrollment_counts));
}

bool EmbeddingClassifier::enroll(const char *label, const float *embedding)
{
    // Find existing class or create new one
    int idx = -1;
    for (int i = 0; i < m_class_count; i++) {
        if (strncmp(m_labels[i], label, 32) == 0) {
            idx = i;
            break;
        }
    }
    if (idx == -1) {
        if (m_class_count >= MAX_CLASSES) return false;
        idx = m_class_count++;
        strncpy(m_labels[idx], label, 31);
        m_labels[idx][31] = '\0';
        m_enrollment_counts[idx] = 0;
    }

    // Accumulate into prototype (we'll normalize on match)
    int n = m_enrollment_counts[idx];
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        // Running average
        m_prototypes[idx][i] = (m_prototypes[idx][i] * n + embedding[i]) / (n + 1);
    }
    m_enrollment_counts[idx]++;
    return true;
}

MatchResult EmbeddingClassifier::match(const float *embedding)
{
    float best_dist = FLT_MAX;
    int   best_idx  = -1;

    for (int i = 0; i < m_class_count; i++) {
        float d = l2distance(embedding, m_prototypes[i]);
        if (d < best_dist) {
            best_dist = d;
            best_idx  = i;
        }
    }

    return {
        .index    = (best_dist <= m_threshold) ? best_idx : -1,
        .distance = best_dist
    };
}

float EmbeddingClassifier::l2distance(const float *a, const float *b)
{
    float sum = 0.0f;
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}