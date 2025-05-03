#ifndef SPECTRA_PARAMS_H
#define SPECTRA_PARAMS_H

typedef struct JONSWAP_params {
    float scale;
    float wind_speed;
    float angle;
    float spread_blend;
    float swell;
    float fetch;
    float depth;
    float short_waves_fade;
    float gamma;
    float peak_omega;
    float alpha;
    float g;
};

#endif