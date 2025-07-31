#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#include <spa/utils/result.h>
#include <spa/param/video/raw.h>
#include <spa/param/video/format-utils.h>

#include <pipewire/pipewire.h>

#include <interface.h>


#define YUY2_BYTES_PER_PIXEL 2
#define DEFAULT_BUFFERS 4
#define MIN_BUFFERS 2
#define MAX_BUFFERS 16

#define MAX_PATH 512

const char* dataPath = "/path/to/cam_infer_models/";
char yoloBasePath[MAX_PATH];
char yoloClassesFile[MAX_PATH];
char yoloModelWeights[MAX_PATH];

uint32_t frame_width;
uint32_t frame_height;
bool detect_done = false;

struct impl {
    struct pw_main_loop *loop;
    struct pw_context *context;
    struct pw_core *core;

    struct pw_stream *capture;
    struct pw_stream *raw_playback;
    struct pw_stream *detection_playback;

    struct spa_hook capture_listener;
    struct spa_hook raw_playback_listener;
    struct spa_hook detection_playback_listener;

    struct spa_video_info_raw capture_info;
    struct spa_video_info_raw raw_playback_info;
    struct spa_video_info_raw detection_playback_info;
};

static void copy_buffer(struct pw_buffer *in, struct pw_buffer *out)
{
    assert(in->buffer->n_datas == out->buffer->n_datas);

    for (uint32_t i = 0; i < out->buffer->n_datas; i++) {
        struct spa_data *src = &in->buffer->datas[i];
        struct spa_data *dst = &out->buffer->datas[i];
        // Copy buffer contents
        memcpy(dst->data, src->data, src->chunk->size);
        dst->chunk->offset = 0;
        dst->chunk->size = src->chunk->size;
        dst->chunk->stride = src->chunk->stride;
    }
}

static void detect_completed_callback (struct pw_buffer* buffer,
                                      struct impl* impl,
                                      bool success)
{
    pw_stream_queue_buffer(impl->detection_playback, buffer);
    pw_stream_trigger_process(impl->detection_playback);
}

static void on_process(void *userdata)
{
    struct impl *impl = userdata;
    struct pw_buffer *in, *out;  // buffers

    // construct paths
    snprintf(yoloBasePath, MAX_PATH, "%syolo/", dataPath);
    snprintf(yoloClassesFile, MAX_PATH, "%scoco.names", yoloBasePath);
    snprintf(yoloModelWeights, MAX_PATH, "%syolo11s.onnx", yoloBasePath);

    float confThreshold = 0.7;
    float nmsThreshold = 0.4;
    bool bVis = true;

    // Add debug logging
    pw_log_debug("on_process called");

    // 1. DEQUEUE phase - Getting buffers from streams
    //out == NULL ⇒ No buffer is available to process/fill — the destination queue is empty.
    if ((in = pw_stream_dequeue_buffer(impl->capture)) == NULL) {
      pw_log_warn("no input buffer");
        return;
    }

    // process raw_playback stream
    //out != NULL ⇒ A buffer is available
    if ((out = pw_stream_dequeue_buffer(impl->raw_playback)) != NULL) {
        pw_log_debug("copying to raw_playback stream");
        copy_buffer(in, out); // src (in) -> dst (out)
        pw_stream_queue_buffer(impl->raw_playback, out);
        pw_stream_trigger_process(impl->raw_playback);
    }

    // process detection_playback stream
    if ((out = pw_stream_dequeue_buffer(impl->detection_playback)) != NULL) {
        pw_log_debug("copying to detection_playback stream");

        copy_buffer(in, out); //detectObjects before copying?
        detectObjects_async(out, confThreshold, nmsThreshold, yoloBasePath,
                           yoloClassesFile, frame_width, frame_height, bVis,
                           detect_completed_callback, impl);
                           
        // detectObjects(out, confThreshold, nmsThreshold, yoloBasePath,
        //   yoloClassesFile, frame_width, frame_height, bVis, &detect_done);
        // pw_stream_queue_buffer(impl->detection_playback, out);
        // pw_stream_trigger_process(impl->detection_playback);
    }

    // return input buffer
    pw_stream_queue_buffer(impl->capture, in);
}


static void on_stream_state_changed(void *userdata, enum pw_stream_state old,
        enum pw_stream_state state, const char *error)
{
    struct impl *impl = userdata;

    printf("stream state changed: %s -> %s\n",
           pw_stream_state_as_string(old),
           pw_stream_state_as_string(state));

    if (error)
        printf(" (error: %s)", error);
    printf("\n");

    if (state == PW_STREAM_STATE_ERROR) {
        pw_main_loop_quit(impl->loop);
    }
}

static void on_capture_param_changed (void *userdata, uint32_t id, const struct spa_pod *param)
{
    struct impl *impl = userdata;
    struct spa_video_info_raw format = SPA_VIDEO_INFO_RAW_INIT(0);
    uint32_t size;
    const struct spa_pod *params[2];
    uint8_t buffer[1024];
    struct spa_pod_builder b;

    if (param == NULL || id != SPA_PARAM_Format)
        return;

    pw_log_info("format changed");
    spa_format_video_raw_parse(param, &format);
    frame_width = format.size.width;
    frame_height = format.size.height;

    size = SPA_ROUND_UP_N(format.size.width * format.size.height * YUY2_BYTES_PER_PIXEL, 4);
    // Configure buffers for the capture stream
    spa_pod_builder_init(&b, buffer, sizeof(buffer));

    params[0] = spa_pod_builder_add_object(&b,
        SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
        SPA_PARAM_BUFFERS_buffers,  SPA_POD_CHOICE_RANGE_Int(DEFAULT_BUFFERS, MIN_BUFFERS, MAX_BUFFERS),
        SPA_PARAM_BUFFERS_blocks,   SPA_POD_Int(1),
        SPA_PARAM_BUFFERS_size,     SPA_POD_Int(size),
        SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(1<<SPA_DATA_MemFd));

    pw_stream_update_params(impl->capture, params, 1);

    // Forward exact format and buffers to the output streams
    spa_pod_builder_init(&b, buffer, sizeof(buffer));

    params[0] = spa_format_video_raw_build(&b, SPA_PARAM_EnumFormat, &format);
    params[1] = spa_pod_builder_add_object(&b,
        SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
        SPA_PARAM_BUFFERS_buffers,  SPA_POD_CHOICE_RANGE_Int(DEFAULT_BUFFERS, MIN_BUFFERS, MAX_BUFFERS),
        SPA_PARAM_BUFFERS_blocks,   SPA_POD_Int(1),
        SPA_PARAM_BUFFERS_size,     SPA_POD_Int(size),
        SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(1<<SPA_DATA_MemFd));

    pw_stream_update_params(impl->raw_playback, params, 2);
    pw_stream_update_params(impl->detection_playback, params, 2);
}

static const struct pw_stream_events capture_stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .process = on_process,
    .param_changed = on_capture_param_changed,
    .state_changed = on_stream_state_changed,
};

static const struct pw_stream_events playback_stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
};

static void signal_handler(int signo)
{
    if (signo == SIGINT) {
        fprintf(stderr, "\nCaught SIGINT, exiting...\n");
        exit(0);
    }
}

int main(int argc, char *argv[])
{
    struct impl *impl;
    const struct spa_pod *params[1];
    uint8_t buffer[1024];
    struct spa_pod_builder b;
    int res;
    spa_autofree char *link_group = NULL;

    impl = calloc(1, sizeof(struct impl));
    if (impl == NULL)
        return -1;

    // Initialize PipeWire
    pw_init(&argc, &argv);

    // Create main loop
    impl->loop = pw_main_loop_new(NULL);
    if (impl->loop == NULL) {
        fprintf(stderr, "Failed to create main loop\n");
        return -1;
    }

    // Create context
    impl->context = pw_context_new(pw_main_loop_get_loop(impl->loop), NULL, 0);
    if (impl->context == NULL) {
        fprintf(stderr, "Failed to create context\n");
        return -1;
    }

    // Connect to PipeWire
    impl->core = pw_context_connect(impl->context, NULL, 0);
    if (impl->core == NULL) {
        fprintf(stderr, "Failed to connect to PipeWire\n");
        return -1;
    }

    // Setup camera format
    impl->capture_info = (struct spa_video_info_raw) {
        .format = SPA_VIDEO_FORMAT_YUY2,
    };
    impl->raw_playback_info = impl->capture_info;
    impl->detection_playback_info = impl->capture_info;

    link_group = spa_aprintf ("camera-filter-%d", getpid());

    // Create capture stream
    impl->capture = pw_stream_new(impl->core, "filter-capture",
            pw_properties_new(
                PW_KEY_MEDIA_TYPE, "Video",
                PW_KEY_MEDIA_CATEGORY, "Capture",
                PW_KEY_MEDIA_ROLE, "Camera",
                PW_KEY_NODE_DESCRIPTION, "camera sink",
		PW_KEY_MEDIA_CLASS, "Stream/Input/Video",
                PW_KEY_NODE_LINK_GROUP, link_group,
                NULL));

    // Create raw playback stream
    impl->raw_playback = pw_stream_new(impl->core, "raw-playback",
            pw_properties_new(
                PW_KEY_MEDIA_TYPE, "Video",
                PW_KEY_MEDIA_CATEGORY, "Playback",
                PW_KEY_MEDIA_ROLE, "Camera",
                PW_KEY_NODE_DESCRIPTION, "raw playback",
                PW_KEY_NODE_NAME, "camera-raw-output",
		PW_KEY_MEDIA_CLASS, "Stream/Output/Video",
                PW_KEY_NODE_LINK_GROUP, link_group,
                NULL));

    // Create Yolo detection playback stream
    impl->detection_playback = pw_stream_new(impl->core, "detection-playback",
            pw_properties_new(
                PW_KEY_MEDIA_TYPE, "Video",
                PW_KEY_MEDIA_CATEGORY, "Playback",
                PW_KEY_MEDIA_ROLE, "Camera",
                PW_KEY_NODE_DESCRIPTION, "detection playback",
                PW_KEY_NODE_NAME, "camera-detection-output",
		PW_KEY_MEDIA_CLASS, "Stream/Output/Video",
                PW_KEY_NODE_LINK_GROUP, link_group,
                NULL));

    // Add listeners
    pw_stream_add_listener(impl->capture,
            &impl->capture_listener,
            &capture_stream_events, impl);
    pw_stream_add_listener(impl->raw_playback,
            &impl->raw_playback_listener,
            &playback_stream_events, impl);
    pw_stream_add_listener(impl->detection_playback,
           &impl->detection_playback_listener,
           &playback_stream_events, impl);

    // Connect streams
    spa_pod_builder_init(&b, buffer, sizeof(buffer));
    params[0] = spa_format_video_raw_build(&b, SPA_PARAM_EnumFormat,
            &impl->capture_info);

    if ((res = pw_stream_connect(impl->capture,
            PW_DIRECTION_INPUT,
            PW_ID_ANY,
            PW_STREAM_FLAG_AUTOCONNECT |
            PW_STREAM_FLAG_MAP_BUFFERS |
            PW_STREAM_FLAG_ASYNC,
            params, 1)) < 0) {
        fprintf(stderr, "Cannot connect capture stream: %s\n", spa_strerror(res));
        return -1;
    }

    spa_pod_builder_init(&b, buffer, sizeof(buffer));
    params[0] = spa_format_video_raw_build(&b, SPA_PARAM_EnumFormat,
            &impl->raw_playback_info);

    if ((res = pw_stream_connect(impl->raw_playback,
            PW_DIRECTION_OUTPUT,
            PW_ID_ANY,
            PW_STREAM_FLAG_MAP_BUFFERS |
            PW_STREAM_FLAG_TRIGGER |
            PW_STREAM_FLAG_ASYNC,
            params, 1)) < 0) {
        fprintf(stderr, "Cannot connect raw playback stream: %s\n", spa_strerror(res));
        return -1;
    }

    spa_pod_builder_init(&b, buffer, sizeof(buffer));
    params[0] = spa_format_video_raw_build(&b, SPA_PARAM_EnumFormat,
            &impl->detection_playback_info);

    if ((res = pw_stream_connect(impl->detection_playback,
            PW_DIRECTION_OUTPUT,
            PW_ID_ANY,
            PW_STREAM_FLAG_MAP_BUFFERS |
            PW_STREAM_FLAG_TRIGGER |
            PW_STREAM_FLAG_ASYNC,
            params, 1)) < 0) {
        fprintf(stderr, "Cannot connect raw playback stream: %s\n", spa_strerror(res));
        return -1;
    }

    // Handle SIGINT
    signal(SIGINT, signal_handler);

    // Run the main loop
    printf("Running... Press Ctrl+C to exit\n");
    pw_main_loop_run(impl->loop);

    // Cleanup
    pw_stream_destroy(impl->capture);
    pw_stream_destroy(impl->raw_playback);
    pw_stream_destroy(impl->detection_playback);
    pw_context_destroy(impl->context);
    pw_main_loop_destroy(impl->loop);
    pw_deinit();
    free(impl);

    return 0;
}
