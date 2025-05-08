/*
 * qpsk_modem.c
 *
 *  Created on: May 8, 2025
 *      Author: adeas
 */

#include "qpsk_modem.h"
#include <math.h>

static const int8_t QPSK_I[4] = {+127,-127,-127,+127};
static const int8_t QPSK_Q[4] = {+127,+127,-127,-127};

static const float PI = 3.14159265358979323846f;
//static const float QPSK_PHASE[4] = { PI/4, 3*PI/4, 5*PI/4, 7*PI/4 };

//QPSK modulation Functions

void QpskModem_Init(QpskModem* modem, uint16_t sample_per_symbol, float f0, float fs){
	modem->samples_per_symbol = sample_per_symbol;
	modem->f0 = f0;
	modem->fs = fs;
}

void QpskModem_Modulate(QpskModem* modem, const uint8_t* data, uint16_t len){
	modem->num_symbols = len*4;
	uint16_t idx = 0;
	for (uint16_t i =0 ; i < len ; i++){
		uint8_t byte = data[i];
		for(int b=6; b>= 0 ; b-=2){
			modem->symbols[idx++] = (byte >> b) & 0x03;
		}
	}
}

void QpskModem_Demodulate(QpskModem* modem, QpskRingBuffer* rxbuf, uint8_t* data_out, uint16_t* len_out) {
    uint16_t nb_symbols = 0;
    uint8_t symbols[QPSK_MAX_SYMBOLS];
    uint32_t global_sample = 0;
    for (uint16_t s = 0; s < QPSK_MAX_SYMBOLS; s++) {
        if (QpskRingBuffer_Available(rxbuf) < modem->samples_per_symbol)
            break;
        float I = 0.0f, Q = 0.0f;
        for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
            float t = (float)global_sample / modem->fs;
            float ref_cos = cosf(2 * M_PI * modem->f0 * t);
            float ref_sin = sinf(2 * M_PI * modem->f0 * t);
            int16_t sample = QpskRingBuffer_Get(rxbuf);
            I += sample * ref_cos;
            Q += sample * ref_sin;
        }
        uint8_t symbol = 0;
        if (I >= 0 && Q >= 0) symbol = 0;
        else if (I < 0 && Q >= 0) symbol = 1;
        else if (I < 0 && Q < 0) symbol = 2;
        else if (I >= 0 && Q < 0) symbol = 3;
        symbols[nb_symbols++] = symbol;
    }
    *len_out = nb_symbols / 4;
    for (uint16_t i = 0; i < *len_out; i++) {
        data_out[i] = (symbols[i*4+0] << 6) | (symbols[i*4+1] << 4) | (symbols[i*4+2] << 2) | (symbols[i*4+3]);
    }
}


void QpskModem_SymbolsToIQ(QpskModem* modem){
	for (uint16_t i=0; i < modem ->num_symbols; i++){
		modem->iq[2*i] = QPSK_I[modem->symbols[i]];
		modem->iq[2*i+1] = QPSK_Q[modem->symbols[i]];
	}
	modem->num_iq = modem->num_symbols*2;
}

void QpskModem_GenerateSignal(QpskModem* modem, QpskRingBuffer* txbuf, float amplitude) {
    uint32_t global_sample = 0;
    for (uint16_t s = 0; s < modem->num_symbols; s++) {
        float I = QPSK_I[modem->symbols[s]] / 127.0f;
        float Q = QPSK_Q[modem->symbols[s]] / 127.0f;
        for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
            float t = (float)global_sample / modem->fs;
            float sample = amplitude * (I * cosf(2 * PI * modem->f0 * t) + Q * sinf(2 * PI * modem->f0 * t));
            QpskRingBuffer_Put(txbuf, (int16_t)(sample * 2047.0f));
        }
    }
}

//Buffer Functions

void QpskRingBuffer_Init(QpskRingBuffer* rb){
	rb->head = 0;
	rb->tail = 0;
}

uint8_t QpskRingBuffer_IsFull(const QpskRingBuffer* rb){
	return ((rb->head + 1) % QPSK_RINGBUF_SIZE) == rb->tail;
}

uint8_t QpskRingBuffer_IsEmpty(const QpskRingBuffer* rb){
	return rb->head == rb->tail;
}

void QpskRingBuffer_Put(QpskRingBuffer* rb, int16_t value) {
    if (!QpskRingBuffer_IsFull(rb)) {
        rb->buf[rb->head] = value;
        rb->head = (rb->head + 1) % QPSK_RINGBUF_SIZE;
    }
}

int16_t QpskRingBuffer_Get(QpskRingBuffer* rb) {
    int16_t val = 0;
    if (!QpskRingBuffer_IsEmpty(rb)) {
        val = rb->buf[rb->tail];
        rb->tail = (rb->tail + 1) % QPSK_RINGBUF_SIZE;
    }
    return val;
}

uint32_t QpskRingBuffer_Available(const QpskRingBuffer* rb) {
    // Returns the number of elements currently in the buffer
    if (rb->head >= rb->tail)
        return rb->head - rb->tail;
    else
        return QPSK_RINGBUF_SIZE - (rb->tail - rb->head);
}


// TESTS FUNCTION

void Qpsk_SimulateReception(QpskRingBuffer* txbuf, QpskRingBuffer* rxbuf) {
    while (!QpskRingBuffer_IsEmpty(txbuf) && !QpskRingBuffer_IsFull(rxbuf)) {
        int16_t val = QpskRingBuffer_Get(txbuf);
        QpskRingBuffer_Put(rxbuf, val);
    }
}
