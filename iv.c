/* Copyright (c) Dmitry "Leo" Kuznetsov 2021 see LICENSE for details */
#include "quick.h"
#include "stb_image.h"
#include "up.h"

// TODO: see README.md

begin_c

static void up2x2(void);

uic_button(upscale, "\xE2\xA7\x88", 0, {
    up2x2();
});

uic_button(full_screen, "\xE2\xA7\x89", 0, {
    app.full_screen(!app.is_full_screen);
});

uic_button(quit, "\xF0\x9F\x9E\xAD", 0, {
    app.close();
});

static_uic_container(buttons, null, &upscale.ui, &full_screen.ui, &quit.ui);

const char* title = "I.V.";

static image_t image;

static char filename[260]; // $(SolutionDir)..\mandrill-4.2.03.png

static void* load_image(const byte* data, int64_t bytes, int32_t* w, int32_t* h,
    int32_t* bpp, int32_t preferred_bytes_per_pixel) {
    void* pixels = stbi_load_from_memory((byte const*)data, (int)bytes, w, h,
        bpp, preferred_bytes_per_pixel);
    return pixels;
}

static void load() {
    int r = 0;
    void* data = null;
    int64_t bytes = 0;
    r = crt.memmap_read(filename, &data, &bytes);
    fatal_if_not_zero(r);
    int w = 0;
    int h = 0;
    int bpp = 0; // bytes (!) per pixel
    void* pixels = load_image(data, bytes, &w, &h, &bpp, 0);
    fatal_if_null(pixels);
    gdi.image_init(&image, w, h, bpp, pixels);
    free(pixels);
    crt.memunmap(data, bytes);
}

static void up2x2(void) {
    const int32_t w = image.w % 2 == 0 ? image.w : image.w - 1;
    const int32_t h = image.h % 2 == 0 ? image.h : image.h - 1;
    up_t u = {
        .input = {
            .p = image.pixels,
            .w = w, // even width
            .h = h, // even height
            .s = image.w * image.bpp,
            .c = image.bpp
        },
        .output = {
            .p = malloc(w * 2 * h * 2 * image.bpp),
            .w = w * 2,
            .h = h * 2,
            .s = w * 2 * image.bpp,
            .c = image.bpp
        },
        .half = { // downscaled image
            .p = (uint8_t*)malloc((w / 2) * (h / 2) * image.bpp),
            .w = w / 2,
            .h = h / 2,
            .s = w / 2 * image.bpp,
            .c = image.bpp
        }
    };
    fatal_if_null(u.output.p);
    fatal_if_null(u.half.p);
    uint64_t seed = crt.nanoseconds();
    #ifdef DEBUG
        seed = 0x1;
    #endif
    up.upscale(&u, seed);
    gdi.image_dispose(&image);
    gdi.image_init(&image, u.output.w, u.output.h, u.output.c, u.output.p);
    free(u.output.p);
    free(u.half.p);
    app.redraw();
}

static void paint(uic_t* ui) {
    gdi.set_brush(gdi.brush_color);
    gdi.set_brush_color(colors.black);
    gdi.fill(0, 0, ui->w, ui->h);
    if (image.w > 0 && image.h > 0) {
        int x = (ui->w - image.w) / 2;
        int y = (ui->h - image.h) / 2;
        gdi.draw_image(x, y, image.w, image.h, &image);
    }
}

static void measure(uic_t* unused(ui)) {
}

static void layout(uic_t* unused(ui)) {
    layouts.horizontal(&buttons, buttons.x, buttons.y, 10);
}

static void openned(void) {
}

static void fini(void) {
    if (image.bitmap != null) {
        gdi.image_dispose(&image);
    }
}

static void init(void) {
    app.title = title;
    app.fini = fini;
    app.openned = openned;
    app.ui->paint = paint;
    app.ui->measure = measure;
    app.ui->layout = layout;
    static uic_t* children[] = { &buttons, null };
    app.ui->children = children;
    //  TODO: load first (depth) image from:
    //  app.known_folder(known_folder_pictures)
    if (app.argc > 1) {
        strprintf(filename, "%s", app.argv[1]);
        if (access(filename, 0) == 0) {
            load();
        }
    }
}

app_t app = {
    .class_name = "iv",
    .init = init,
    .min_width = 640,
    .min_height = 640
};

end_c
