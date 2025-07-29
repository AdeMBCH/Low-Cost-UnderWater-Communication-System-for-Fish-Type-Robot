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

#define BURST_LEN 4
void AskModem_Modulate_DiracTransitions(UART_HandleTypeDef* huart,AskModem* modem,const uint8_t* payload,uint16_t byte_len,AskRingBuffer* outbuf,float amplitude)
{
    uint32_t global_sample = 0;
    int16_t preamble_high = (int16_t)(amplitude * 1024.0f);
    int16_t preamble_low = 0;

    // === Préambule : 1 bit à 1 ===
    for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
    	if (n < BURST_LEN || n >= modem->samples_per_symbol - BURST_LEN){
        	AskRingBuffer_Put(outbuf, preamble_high);
        }else{
        	AskRingBuffer_Put(outbuf, preamble_low);
        }
    }

    // === Préambule : 1 bit à 0 ===
    for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
        AskRingBuffer_Put(outbuf, preamble_low);
    }

    // === Données utiles ===
    for (uint16_t i = 0; i < byte_len; i++) {
        uint8_t byte = payload[i];
        for (int b = 7; b >= 0; b--) {
            uint8_t bit = (byte >> b) & 1;

            // Chercher bit précédent et suivant (si disponibles)
            uint8_t prev_bit = 0;
            uint8_t next_bit = 0;

            if (i > 0 || b < 7) {
                uint8_t prev_byte = (b == 7) ? payload[i - 1] : byte;
                int prev_b = (b == 7) ? 0 : b + 1;
                prev_bit = (prev_byte >> prev_b) & 1;
            }

            if (i < byte_len - 1 || b > 0) {
                uint8_t next_byte = (b == 0) ? payload[i + 1] : byte;
                int next_b = (b == 0) ? 7 : b - 1;
                next_bit = (next_byte >> next_b) & 1;
            }

            for (uint16_t n = 0; n < modem->samples_per_symbol; n++, global_sample++) {
                if (bit) {
                    // Début de bloc de 1
                    if (prev_bit == 0 && n < BURST_LEN) {
                        AskRingBuffer_Put(outbuf, preamble_high);
                    }
                    // Fin de bloc de 1
                    else if (next_bit == 0 && n >= modem->samples_per_symbol - BURST_LEN) {
                        AskRingBuffer_Put(outbuf, preamble_high);
                    }
                    else {
                        AskRingBuffer_Put(outbuf, preamble_low);
                    }
                } else {
                    AskRingBuffer_Put(outbuf, preamble_low);
                }
            }
        }
    }
}

#define OOK_THRESHOLD 650
#define MIN_RUN_LENGTH 100

void AskModem_Demodulate_ByEdges(UART_HandleTypeDef* huart, uint16_t* adc_buffer, uint16_t buffer_len, uint8_t* bits_out, uint16_t* bit_len)
{
    *bit_len = 0;
    if (buffer_len < 3) return;

    // 1. Binariser le signal en 0 / 1 (valeurs ADC => logique binaire)
    uint16_t binary[buffer_len];
    uint16_t k = 0;
    while (k < buffer_len) {
        if (adc_buffer[k] > OOK_THRESHOLD) {
            // Début d’un run de 1
            uint16_t start = k;
            while (k < buffer_len && adc_buffer[k] > OOK_THRESHOLD) k++;
            uint16_t run_len = k - start;

            uint16_t value = (run_len >= MIN_RUN_LENGTH) ? 1 : 0;
            for (uint16_t j = start; j < k; j++) {
                binary[j] = value;
            }
        } else {
            binary[k++] = 0;
        }
    }

    // 2. Repérer les blocs de 1 consécutifs et enregistrer leur centre
    uint16_t peaks[ASK_MAX_BITS * 2] = {0};
    uint16_t num_peaks = 0;

    uint16_t i = 0;
    while (i < buffer_len) {
        if (binary[i] == 1) {
            uint16_t start = i;
            while (i < buffer_len && binary[i] == 1) i++;
            uint16_t end = i - 1;
            uint16_t center = (start + end) / 2;

            if (num_peaks < (ASK_MAX_BITS * 2)) {
                peaks[num_peaks++] = center;
            }
        } else {
            i++;
        }
    }

    if (num_peaks < 2) return;

    // 3. Estimer la durée d’un bit
    uint16_t bit_unit_len = peaks[1] - peaks[0];
    if (bit_unit_len == 0) bit_unit_len = 1;

    // 4. Ajouter un pic fictif si impair
    if ((num_peaks % 2) != 0 && num_peaks < (ASK_MAX_BITS * 2)) {
        peaks[num_peaks] = peaks[num_peaks - 1] + bit_unit_len;
        num_peaks++;
    }
    uint8_t current_bit = 1;
    for (uint16_t j = 0; j + 1 < num_peaks && *bit_len < ASK_MAX_BITS; j++) {
        uint16_t delta = peaks[j + 1] - peaks[j];
        uint16_t bit_count = (delta + bit_unit_len / 2) / bit_unit_len;

        if (bit_count == 0) bit_count = 1;
        if (bit_count > 8) bit_count = 8;  // or 10 max
        for (uint16_t b = 0; b < bit_count && *bit_len < ASK_MAX_BITS; b++) {
            bits_out[(*bit_len)++] = current_bit;
        }

        current_bit = !current_bit;
    }
}


void AskModem_Demodulate_OOK(UART_HandleTypeDef* huart,uint16_t* adc_buffer,uint16_t buffer_len,uint8_t* bits_out,uint16_t* bit_len)
{
    *bit_len = 0;

    // 1. Convertir en binaire brut (0/1)
    uint8_t binary_stream[buffer_len];
    for (uint16_t i = 0; i < buffer_len; i++) {
        binary_stream[i] = (adc_buffer[i] > OOK_THRESHOLD) ? 1 : 0;
    }

    // 2. Trouver la longueur du premier bloc de '1'
    uint16_t i = 0;
    while (i < buffer_len && binary_stream[i] == 0) i++;  // skip 0s
    if (i == buffer_len) return; // pas de 1 détecté

    uint16_t bit_unit_length = 0;
    uint8_t first_bit_value = binary_stream[i];

    while (i + bit_unit_length < buffer_len && binary_stream[i + bit_unit_length] == first_bit_value) {
        bit_unit_length++;
    }

    if (bit_unit_length < 3) return;  // bloc trop court = bruit ?

    // 3. Parcourir le reste et regrouper par multiple de bit_unit_length
    while (i < buffer_len && *bit_len < ASK_MAX_BITS)
    {
        uint8_t current = binary_stream[i];
        uint16_t run_len = 0;
        while ((i + run_len < buffer_len) && (binary_stream[i + run_len] == current)) {
            run_len++;
        }

        // Nombre de bits ≈ run_len / bit_unit_length (arrondi)
        uint16_t num_bits = (run_len + bit_unit_length / 2) / bit_unit_length;

        for (uint16_t b = 0; b < num_bits && *bit_len < ASK_MAX_BITS; b++) {
            bits_out[(*bit_len)++] = current;
        }

        i += run_len;
    }
}

/*
void AskModem_Demodulate_OOK(UART_HandleTypeDef* huart, uint16_t* adc_buffer, uint16_t buffer_len, uint16_t samples_per_symbol, uint8_t* bits_out, uint16_t* bit_len)
{
    *bit_len = 0;
    uint16_t num_symbols = buffer_len / samples_per_symbol;

    for (uint16_t sym = 0; sym < num_symbols && *bit_len < ASK_MAX_BITS; sym++)
    {
        uint16_t high_count = 0;

        for (uint16_t i = 0; i < samples_per_symbol; i++)
        {
            uint16_t idx = sym * samples_per_symbol + i;
            if (adc_buffer[idx] > OOK_THRESHOLD)
                high_count++;
        }

        uint8_t bit = (high_count > (samples_per_symbol / 2)) ? 1 : 0;
        bits_out[(*bit_len)++] = bit;
    }
}*/

/*
void AskModem_Demodulate_OOK(UART_HandleTypeDef* huart, AskModem* modem, AskRingBuffer* inbuf, uint8_t* bits_out, uint16_t* bit_len)
{
    *bit_len = 0;
    const uint16_t N = modem->samples_per_symbol;

    if (N == 0 || AskRingBuffer_Available(inbuf) < 2 * N)
        return;

    while (AskRingBuffer_Available(inbuf) >= N && *bit_len < ASK_MAX_BITS)
    {
        uint32_t sum = 0;
        for (uint16_t i = 0; i < N; ++i) {
            sum += AskRingBuffer_Get(inbuf);
        }

        float avg = (float)sum / N;

        bits_out[(*bit_len)++] = (avg >= OOK_THRESHOLD) ? 1 : 0;
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
