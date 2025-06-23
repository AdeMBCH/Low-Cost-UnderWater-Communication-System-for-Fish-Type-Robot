/*
 * iq_transmitter.h
 *
 *  Created on: Jun 3, 2025
 *      Author: adeas
 */

#ifndef INC_IQ_TRANSMITTER_H_
#define INC_IQ_TRANSMITTER_H_

#include "stm32f4xx_hal.h"
#include "qpsk_modem.h"

typedef struct {
    uint16_t *buffer;
    uint32_t length;
    volatile uint32_t index;
    volatile uint8_t active;
} IQTransmitter;

// Initialise l’émetteur avec un buffer de signal mono modulé (ex: généré par QpskModem_GenerateSignal)
void IQTransmitter_InitFromBuffer(const int16_t* signal, uint16_t length);

// Démarre la transmission via SPI + timer
void IQTransmitter_Start(void);

// Arrête la transmission en cours
void IQTransmitter_Stop(void);

void Enable_TIM2_Interrupt(void);

#endif /* INC_IQ_TRANSMITTER_H_ */
