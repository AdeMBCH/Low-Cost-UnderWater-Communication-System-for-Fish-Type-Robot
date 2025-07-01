#include "ask_modem.h"
#include <math.h>
#include "uart_protocol.h"
#include <stdlib.h>

void AskModem_Init(AskModem* modem, uint16_t sample_per_symbol, float f0, float fs) {
    modem->samples_per_symbol = sample_per_symbol;
    modem->f0 = f0;
    modem->fs = fs;
}

void AskModem_Modulate(UART_HandleTypeDef* huart, AskModem* modem, const uint8_t* bits, uint16_t bit_len, AskRingBuffer* outbuf, float amplitude) {
    uint32_t global_sample = 0;
    for (uint16_t i = 0; i < bit_len; i++) {
        uint8_t byte = bits[i];
        for (int b = 7; b >= 0; b--) {
            uint8_t bit = (byte >> b) & 1;
            for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
                float t = (float)global_sample / modem->fs;
                float carrier = cosf(2 * M_PI * modem->f0 * t);
                float sample = bit ? (amplitude * carrier) : 0.0f;
                AskRingBuffer_Put(outbuf, (int16_t)(sample * 2047.0f));
            }
        }
    }
}
void AskModem_Modulate_OOK(UART_HandleTypeDef* huart, AskModem* modem, const uint8_t* payload, uint16_t byte_len, AskRingBuffer* outbuf, float amplitude) {
    uint32_t global_sample = 0;

    // === Préambule : 1 bit à 1 ===
    int16_t preamble_high = (int16_t)(amplitude * 1024.0f);
    for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
        AskRingBuffer_Put(outbuf, preamble_high);
    }

    // === Préambule : 1 bit à 0 ===
    int16_t preamble_low = 0;
    for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
        AskRingBuffer_Put(outbuf, preamble_low);
    }

    // === Données utiles ===
    for (uint16_t i = 0; i < byte_len; i++) {
        uint8_t byte = payload[i];
        for (int b = 7; b >= 0; b--) {
            uint8_t bit = (byte >> b) & 1;
            int16_t sample = bit ? preamble_high : preamble_low;

            for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
                AskRingBuffer_Put(outbuf, sample);
            }
        }
    }
}

void AskModem_Demodulate_OOK(UART_HandleTypeDef* huart, AskModem* modem, AskRingBuffer* inbuf, uint8_t* bits_out, uint16_t* bit_len) {
    *bit_len = 0;
    const uint16_t N = modem->samples_per_symbol;

    if (AskRingBuffer_Available(inbuf) < 2 * N)
        return;

    // Dynamique selon amplitude connue (3.3V * 1024 = ~3379)
    float threshold = 1800.0f;

    int synced = 0;
    while (!synced && AskRingBuffer_Available(inbuf) >= 2 * N) {
        float avg1 = 0.0f, avg2 = 0.0f;

        for (uint16_t i = 0; i < N; i++) avg1 += fabsf((float)AskRingBuffer_Get(inbuf));
        avg1 /= N;

        for (uint16_t i = 0; i < N; i++) avg2 += fabsf((float)AskRingBuffer_Get(inbuf));
        avg2 /= N;

        if (avg1 > threshold && avg2 < threshold) {
            synced = 1;
        }
    }

    // Lecture des symboles après synchro
    while (AskRingBuffer_Available(inbuf) >= N && *bit_len < ASK_MAX_BITS) {
        float energy = 0.0f;
        for (uint16_t i = 0; i < N; ++i) {
            energy += fabsf((float)AskRingBuffer_Get(inbuf));
        }

        float avg = energy / N;
        bits_out[(*bit_len)++] = (avg > threshold) ? 1 : 0;
    }
}

/*
void AskModem_Modulate_OOK(UART_HandleTypeDef* huart, AskModem* modem, const uint8_t* payload, uint16_t byte_len, AskRingBuffer* outbuf, float amplitude) {
    uint32_t global_sample = 0;

    for (uint16_t i = 0; i < byte_len; i++) {
        uint8_t byte = payload[i];
        for (int b = 7; b >= 0; b--) {
            uint8_t bit = (byte >> b) & 1;
            int16_t sample = bit ? (int16_t)(amplitude * 1024.0f) : 0;

            for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
                AskRingBuffer_Put(outbuf, sample);
            }
        }
    }
}*/
/*
void AskModem_Modulate(UART_HandleTypeDef* huart, AskModem* modem, const uint8_t* bits, uint16_t bit_len, AskRingBuffer* outbuf, float amplitude) {
    uint32_t global_sample = 0;
    for (uint16_t i = 0; i < bit_len; i++) {
        uint8_t bit = bits[i];
        for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
            float t = (float)global_sample / modem->fs;
            float carrier = cosf(2 * M_PI * modem->f0 * t);
            float sample = bit ? (amplitude * carrier) : 0.0f;
            AskRingBuffer_Put(outbuf, (int16_t)(sample * 2047.0f));
        }
    }
}*/
/*
void AskModem_Demodulate(UART_HandleTypeDef* huart, AskModem* modem, AskRingBuffer* inbuf, uint8_t* bits_out, uint16_t* bit_len) {
    *bit_len = 0;

    const uint16_t N = modem->samples_per_symbol;

    while (AskRingBuffer_Available(inbuf) >= N) {
        float energy = 0.0f;

        for (uint16_t i = 0; i < N; ++i) {
            int16_t sample = AskRingBuffer_Get(inbuf);
            energy += fabsf((float)sample);
        }

        float average_energy = energy / N;
        uint8_t bit = (average_energy > 1000.0f) ? 1 : 0;  // 625 à 715 typique

        bits_out[(*bit_len)++] = bit;
    }
}*/

//SIGNAL DETECTED DESTRUCITF
/*
uint8_t SignalDetected(AskRingBuffer* buf, uint32_t threshold) {
    uint32_t energy = 0;
    if (AskRingBuffer_Available(buf) < 64) return 0;

    for (int i = 0; i < 64; i++) {
        int16_t s = AskRingBuffer_Get(buf);
        int16_t centered = s - 2048;  // 👈 important
        energy += abs(centered);
    }

    return (energy > threshold) ? 1 : 0;
}*/

uint8_t SignalDetected(AskRingBuffer* buf, uint32_t threshold) {
    uint32_t energy = 0;
    if (AskRingBuffer_Available(buf) < 64) return 0;

    // On fait une copie temporaire du tail et on lit sans consommer
    uint16_t original_tail = buf->tail;

    for (int i = 0; i < 64; i++) {
        int16_t s = buf->buf[buf->tail];
        buf->tail = (buf->tail + 1) % ASK_RINGBUF_SIZE;

        int16_t centered = s - 2048;
        energy += abs(centered);
    }

    buf->tail = original_tail;  // restore l'état initial (non destructif)

    return (energy > threshold) ? 1 : 0;
}

void AskRingBuffer_Init(AskRingBuffer* rb) {
    rb->head = rb->tail = 0;
}

uint8_t AskRingBuffer_IsFull(const AskRingBuffer* rb) {
    return ((rb->head + 1) % ASK_RINGBUF_SIZE) == rb->tail;
}

uint8_t AskRingBuffer_IsEmpty(const AskRingBuffer* rb) {
    return rb->head == rb->tail;
}

void AskRingBuffer_Put(AskRingBuffer* rb, int16_t value) {
    if (!AskRingBuffer_IsFull(rb)) {
        rb->buf[rb->head] = value;
        rb->head = (rb->head + 1) % ASK_RINGBUF_SIZE;
    }
}

int16_t AskRingBuffer_Get(AskRingBuffer* rb) {
    int16_t val = 0;
    if (!AskRingBuffer_IsEmpty(rb)) {
        val = rb->buf[rb->tail];
        rb->tail = (rb->tail + 1) % ASK_RINGBUF_SIZE;
    }
    return val;
}

uint32_t AskRingBuffer_Available(const AskRingBuffer* rb) {
    if (rb->head >= rb->tail)
        return rb->head - rb->tail;
    else
        return ASK_RINGBUF_SIZE - (rb->tail - rb->head);
}
