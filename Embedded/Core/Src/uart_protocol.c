/*
 * uart_protocol.c
 *
 *  Created on: May 7, 2025
 *      Author: adeas
 */


#include "uart_protocol.h"
#include "stm32f4xx_hal.h"
#include "CMD.h"

static uint8_t CalcChecksum(uint16_t cmd, uint16_t len, uint8_t* payload) {
    uint8_t cs = 0;
    cs ^= 0xFE;
    cs ^= (cmd >> 8) & 0xFF;
    cs ^= (cmd >> 0) & 0xFF;
    cs ^= (len >> 8) & 0xFF;
    cs ^= (len >> 0) & 0xFF;
    for (uint16_t i = 0; i < len; i++) {
        cs ^= payload[i];
    }
    return cs;
}

void UartProtocol_Init(UartProtocol* proto, FrameReceivedCallback cb) {
    proto->rx_state = WAIT_SOF;
    proto->rx_cmd = 0;
    proto->rx_len = 0;
    proto->rx_payload_idx = 0;
    proto->rx_checksum = 0;
    proto->rx_calc_checksum = 0;
    proto->onFrameReceived = cb;
}

void UartProtocol_ParseByte(UartProtocol* proto, uint8_t c) {
    switch(proto->rx_state) {
        case WAIT_SOF:
            if (c == 0xFE) {
                proto->rx_state = READ_CMD_MSB;
                proto->rx_calc_checksum = 0xFE;
            }
            break;
        case READ_CMD_MSB:
            proto->rx_cmd = ((uint16_t)c) << 8;
            proto->rx_calc_checksum ^= c;
            proto->rx_state = READ_CMD_LSB;
            break;
        case READ_CMD_LSB:
            proto->rx_cmd |= c;
            proto->rx_calc_checksum ^= c;
            proto->rx_state = READ_LEN_MSB;
            break;
        case READ_LEN_MSB:
            proto->rx_len = ((uint16_t)c) << 8;
            proto->rx_calc_checksum ^= c;
            proto->rx_state = READ_LEN_LSB;
            break;
        case READ_LEN_LSB:
            proto->rx_len |= c;
            proto->rx_calc_checksum ^= c;
            if (proto->rx_len > FRAME_MAX_PAYLOAD) {
                proto->rx_state = WAIT_SOF;
            } else if (proto->rx_len == 0) {
                proto->rx_state = READ_CHECKSUM;
            } else {
                proto->rx_payload_idx = 0;
                proto->rx_state = READ_PAYLOAD;
            }
            break;
        case READ_PAYLOAD:
            proto->frame_payload[proto->rx_payload_idx++] = c;
            proto->rx_calc_checksum ^= c;
            if (proto->rx_payload_idx >= proto->rx_len) {
                proto->rx_state = READ_CHECKSUM;
            }
            break;
        case READ_CHECKSUM:
            proto->rx_checksum = c;
            if (proto->rx_checksum == proto->rx_calc_checksum && proto->onFrameReceived) {
                proto->onFrameReceived(proto, proto->rx_cmd, proto->rx_len, proto->frame_payload);
            }
            proto->rx_state = WAIT_SOF;
            break;
        default:
            proto->rx_state = WAIT_SOF;
            break;
    }
}

void UartProtocol_SendFrame(UART_HandleTypeDef* huart, uint16_t cmd, uint16_t len, uint8_t* payload) {
    uint8_t tx_buf[6 + FRAME_MAX_PAYLOAD];
    int pos = 0;
    tx_buf[pos++] = 0xFE;
    tx_buf[pos++] = (cmd >> 8) & 0xFF;
    tx_buf[pos++] = (cmd >> 0) & 0xFF;
    tx_buf[pos++] = (len >> 8) & 0xFF;
    tx_buf[pos++] = (len >> 0) & 0xFF;
    for (int i = 0; i < len; i++) {
        tx_buf[pos++] = payload[i];
    }
    uint8_t cs = CalcChecksum(cmd, len, payload);
    tx_buf[pos++] = cs;
    HAL_UART_Transmit(huart, tx_buf, pos, 100);
}

void SendIQFrame(UART_HandleTypeDef* huart, int8_t i, int8_t q) {
    uint8_t payload[3];
    payload[0] = 'T'; // Pour TX
    payload[1] = (uint8_t)i;
    payload[2] = (uint8_t)q;
    UartProtocol_SendFrame(huart, CMD_IQ_DATA, 3, payload);
}
