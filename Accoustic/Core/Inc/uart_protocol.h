/*
 * uart_protocol.h
 *
 *  Created on: May 7, 2025
 *      Author: adeas
 */

#ifndef INC_UART_PROTOCOL_H_
#define INC_UART_PROTOCOL_H_

#include <stdint.h>
#include "stm32f4xx_hal.h"

#define FRAME_MAX_PAYLOAD 256

typedef struct UartProtocol UartProtocol;

typedef void (*FrameReceivedCallback)(UartProtocol* proto, uint16_t cmd, uint16_t len, uint8_t* payload);

struct UartProtocol {
    enum {
        WAIT_SOF,
        READ_CMD_MSB,
        READ_CMD_LSB,
        READ_LEN_MSB,
        READ_LEN_LSB,
        READ_PAYLOAD,
        READ_CHECKSUM
    } rx_state;

    uint16_t rx_cmd;
    uint16_t rx_len;
    uint16_t rx_payload_idx;
    uint8_t rx_checksum;
    uint8_t rx_calc_checksum;
    uint8_t frame_payload[FRAME_MAX_PAYLOAD];

    FrameReceivedCallback onFrameReceived;
};


void UartProtocol_Init(UartProtocol* proto, FrameReceivedCallback cb);

void UartProtocol_ParseByte(UartProtocol* proto, uint8_t c);

void UartProtocol_SendFrame(UART_HandleTypeDef* huart, uint16_t cmd, uint16_t len, uint8_t* payload);

int UartProtocol_BuildFrame(uint16_t cmd, uint16_t len, uint8_t* payload, uint8_t* out_buf);

void SendIQFrame(UART_HandleTypeDef* huart, int8_t i, int8_t q);

uint16_t SyncAndDecodeBits(uint8_t* bits, uint16_t len, uint8_t* chars_out, uint16_t max_chars);

#endif /* INC_UART_PROTOCOL_H_ */
