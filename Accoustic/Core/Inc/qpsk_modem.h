/*
 * qpsk_modem.h
 *
 *  Created on: May 8, 2025
 *      Author: adeas
 */

#ifndef QPSK_MODEM_H
#define QPSK_MODEM_H

#include <stdint.h>
#include "stm32f4xx_hal.h"

#define QPSK_MAX_SYMBOLS 256
#define QPSK_RINGBUF_SIZE (QPSK_MAX_SYMBOLS * 16) // 16 samples

typedef struct {
    uint8_t symbols[QPSK_MAX_SYMBOLS];
    uint16_t num_symbols;
    int8_t iq[2 * QPSK_MAX_SYMBOLS]; //[I0,Q0,I1,Q1,...]
    uint16_t num_iq;
    uint16_t samples_per_symbol;
    float f0;
    float fs;
} QpskModem;

typedef struct {
	int16_t buf[QPSK_RINGBUF_SIZE];
	volatile uint32_t head;
	volatile uint32_t tail;
} QpskRingBuffer;

//QPSK modulation Functions

void QpskModem_Init(QpskModem* modem, uint16_t sample_per_symbol, float f0, float fs);
void QpskModem_Modulate(UART_HandleTypeDef* huart2, QpskModem* modem, const uint8_t* data, uint16_t len);
void QpskModem_Demodulate(UART_HandleTypeDef* huart2,QpskModem* modem, QpskRingBuffer* rxbuf, uint8_t* data_out, uint16_t* len_out);
void QpskModem_SymbolsToIQ(QpskModem* modem);
void QpskModem_GenerateSignal(QpskModem* modem, QpskRingBuffer* txbuf, float amplitude);
void Qpsk_SimulateReception(QpskRingBuffer* txbuf, QpskRingBuffer* rxbuf);

//Buffer Functions

void QpskRingBuffer_Init(QpskRingBuffer* rb);
uint8_t QpskRingBuffer_IsFull(const QpskRingBuffer* rb);
uint8_t QpskRingBuffer_IsEmpty(const QpskRingBuffer* rb);
void QpskRingBuffer_Put(QpskRingBuffer* rb, int16_t value);
int16_t QpskRingBuffer_Get(QpskRingBuffer* rb);
uint32_t QpskRingBuffer_Available(const QpskRingBuffer* rb);

#endif
