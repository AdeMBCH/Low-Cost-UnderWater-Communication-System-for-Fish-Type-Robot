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
        uint8_t bit = bits[i];
        for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
            float t = (float)global_sample / modem->fs;
            float carrier = cosf(2 * M_PI * modem->f0 * t);
            float sample = bit ? (amplitude * carrier) : 0.0f;
            AskRingBuffer_Put(outbuf, (int16_t)(sample * 2047.0f));
        }
    }
}

void AskModem_Demodulate(UART_HandleTypeDef* huart, AskModem* modem, AskRingBuffer* inbuf, uint8_t* bits_out, uint16_t* bit_len) {
    *bit_len = 0;

    while (AskRingBuffer_Available(inbuf) >= modem->samples_per_symbol) {
        uint32_t energy = 0;

        for (uint16_t i = 0; i < modem->samples_per_symbol; i++) {
            int16_t sample = AskRingBuffer_Get(inbuf);
            energy += abs(sample);
        }

        uint32_t threshold = modem->samples_per_symbol * 500;

        bits_out[*bit_len] = (energy > threshold) ? 1 : 0;
        (*bit_len)++;
    }
}

/*
void AskModem_Demodulate(UART_HandleTypeDef* huart, AskModem* modem, AskRingBuffer* inbuf, uint8_t* bits_out, uint16_t* bit_len) {
	uint32_t global_sample = 0;
	*bit_len = 0;

	while (AskRingBuffer_Available(inbuf) >= modem->samples_per_symbol) {
	    float sum = 0.0f;

	    for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
	        int16_t sample = AskRingBuffer_Get(inbuf);

	        // Ici on ne recentre pas, car la tension 0 correspond à "pas de porteuse"
	        sum += sample;
	    }

	    float avg = sum / modem->samples_per_symbol;

	    // ⚠️ Le seuil dépend de ton Vmax mesuré (~2.5V sur 12 bits = ~3100)
	    bits_out[*bit_len] = (avg > 2200.0f) ? 1 : 0;
	    (*bit_len)++;
	}
}*/

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
