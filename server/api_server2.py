"""视频场景分割API服务器

该模块提供了一个基于Flask的RESTful API服务，用于处理视频场景分割任务。
主要功能包括：
- 接收视频文件路径
- 进行场景分割处理
- 返回分割结果，包括每个场景的起始帧和时间戳

依赖项：
- Flask: Web框架
- OpenCV (cv2): 视频处理
- scene_detection: 自定义场景分割模块

作者: MediaSymphony Team
日期: 2024-02
"""

from flask import Flask, request, jsonify
import os
import cv2
from core.scene_detection import SceneDetector
from werkzeug.utils import secure_filename
from utils.logger import Logger
from moviepy import VideoFileClip
import threading
from functools import partial
import time
import traceback
import sys
import tensorflow as tf
import psutil
import nvidia_smi

# 初始化GPU监控
nvidia_smi.nvmlInit()


# 资源管理器类
class ResourceManager:
    # 初始化
    def __init__(self, max_gpu_memory=10, max_cpu_memory=40):
        self.max_gpu_memory = max_gpu_memory  # GPU最大内存(GB)
        self.max_cpu_memory = max_cpu_memory  # CPU最大内存(GB)
        self.logger = Logger("resource_manager")
        self.using_gpu = False
        self.gpu_device = None
        self.last_check_time = 0
        self.check_interval = 5  # 资源检查间隔（秒）

    # 检查GPU是否可用
    def check_gpu_availability(self):
        try:
            physical_devices = tf.config.list_physical_devices("GPU")

            if not physical_devices:
                return False, None, 0

            # 获取所有GPU的信息
            device_count = nvidia_smi.nvmlDeviceGetCount()
            max_free_memory = 0
            selected_device = None

            for i in range(device_count):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                free_memory = info.free / 1024**3  # 转换为GB

                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    selected_device = f"/GPU:{i}"

            if selected_device and max_free_memory > 1:
                return True, selected_device, max_free_memory

            return False, None, 0

        except Exception as e:
            self.logger.error(f"GPU检查失败: {str(e)}")
            return False, None, 0

    # 设置GPU内存增长策略
    def setup_gpu_memory_growth(self):
        try:
            physical_devices = tf.config.list_physical_devices("GPU")

            if physical_devices:
                for device in physical_devices:
                    # 启用内存增长
                    tf.config.experimental.set_memory_growth(device, True)

                    # 设置显存限制
                    memory_limit = int(self.max_gpu_memory * 1024)  # 转换为MB
                    tf.config.set_logical_device_configuration(
                        device,
                        [
                            tf.config.LogicalDeviceConfiguration(
                                memory_limit=memory_limit
                            )
                        ],
                    )

                self.logger.info(f"GPU内存限制设置成功: {self.max_gpu_memory}GB")
        except Exception as e:
            self.logger.error(f"设置GPU内存限制失败: {str(e)}")

    # 监控内存使用情况
    def monitor_memory_usage(self):
        current_time = time.time()
        
        # 检查是否需要进行资源监控
        if current_time - self.last_check_time < self.check_interval:
            return None, None
            
        self.last_check_time = current_time
        
        try:
            # 监控CPU内存
            memory = psutil.virtual_memory()
            cpu_percent = memory.percent
            cpu_used_gb = memory.used / 1024**3

            if cpu_used_gb > self.max_cpu_memory:
                self.logger.warning(f"CPU内存使用接近限制: {cpu_used_gb:.2f}GB/{self.max_cpu_memory}GB")
                self.cleanup_memory()

            # 监控GPU内存
            if self.using_gpu and self.gpu_device:
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(
                    int(self.gpu_device.split(":")[1])
                )
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                gpu_used_gb = info.used / 1024**3

                if gpu_used_gb > self.max_gpu_memory * 0.9:  # 90%警戒线
                    self.logger.warning(f"GPU内存使用接近限制: {gpu_used_gb:.2f}GB/{self.max_gpu_memory}GB")
                    self.cleanup_memory()

            return cpu_used_gb, cpu_percent
            
        except Exception as e:
            self.logger.error(f"监控资源使用时发生错误: {str(e)}")
            return None, None
            
    def cleanup_memory(self):
        """清理系统内存"""
        try:
            # 清理Python垃圾回收
            import gc
            gc.collect()
            
            # 清理GPU缓存
            if self.using_gpu:
                tf.keras.backend.clear_session()
                
            # 清理系统缓存
            if hasattr(psutil, "Process"):
                current_process = psutil.Process()
                current_process.memory_info()
                
            self.logger.info("内存清理完成")
            
        except Exception as e:
            self.logger.error(f"清理内存时发生错误: {str(e)}")
            
    def __del__(self):
        """析构函数，确保资源正确释放"""
        try:
            if self.using_gpu:
                tf.keras.backend.clear_session()
            nvidia_smi.nvmlShutdown()
        except Exception as e:
            self.logger.error(f"释放资源时发生错误: {str(e)}")


# 增强版场景检测器类
class EnhancedSceneDetector(SceneDetector):
    # 初始化
    def __init__(self, resource_manager, logger=None):
        super().__init__(logger=logger)
        self.resource_manager = resource_manager
        self.setup_device()

    # 设置计算设备
    def setup_device(self):
        self.resource_manager.setup_gpu_memory_growth()
        is_gpu_available, device_name, free_memory = (
            self.resource_manager.check_gpu_availability()
        )

        if is_gpu_available:
            self.device_name = device_name
            self.resource_manager.using_gpu = True
            self.resource_manager.gpu_device = device_name
            self.logger.info(
                f"使用GPU设备 {device_name}，可用显存: {free_memory:.2f}GB"
            )

            # 设置混合精度
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        else:
            self.device_name = "/CPU:0"
            self.resource_manager.using_gpu = False
            self.logger.info("使用CPU处理")

    @tf.function
    def process_batch(self, batch_frames):
        """使用tf.function加速批处理"""
        return self.model(batch_frames, training=False)

    def predict_video(self, input_path):
        """重写预测方法，添加GPU支持和内存监控"""
        batch_size = 32 if self.resource_manager.using_gpu else 16

        try:
            video_frames = []
            scenes = []
            single_frame_predictions = []
            all_frame_predictions = []

            cap = cv2.VideoCapture(input_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with tf.device(self.device_name):
                for i in range(0, total_frames, batch_size):
                    # 监控内存使用
                    cpu_used, cpu_percent = self.resource_manager.monitor_memory_usage()

                    # 处理当前批次
                    batch_frames = []
                    for j in range(batch_size):
                        if i + j < total_frames:
                            ret, frame = cap.read()
                            if ret:
                                processed_frame = self.preprocess_frame(frame)
                                batch_frames.append(processed_frame)
                                video_frames.append(frame)

                    if batch_frames:
                        batch_tensor = tf.stack(batch_frames)
                        batch_predictions = self.process_batch(batch_tensor)

                        # 处理预测结果
                        single_frame_predictions.extend(
                            self.process_predictions(batch_predictions)
                        )
                        all_frame_predictions.extend(batch_predictions.numpy())

                        # 清理内存
                        tf.keras.backend.clear_session()

            cap.release()

            # 场景检测
            scenes = self.predictions_to_scenes(single_frame_predictions)

            return video_frames, scenes, single_frame_predictions, all_frame_predictions

        except Exception as e:
            self.logger.error(f"视频预测失败: {str(e)}")
            raise


class AudioMode:
    """音频处理模式"""

    MUTE = "mute"  # 静音模式
    UNMUTE = "un-mute"  # 非静音模式


app = Flask(__name__)
logger = Logger("scene_detection_api")

# 创建资源管理器
resource_manager = ResourceManager(max_gpu_memory=10, max_cpu_memory=40)

# 模型实例和锁
detector_model = None
model_lock = threading.Lock()

SCENE_DETECTION_TIMEOUT = 1800
VIDEO_CODEC = "h264_nvenc"
ALLOWED_EXTENSIONS = {
    ext.split("/")[-1] for ext in ["video/mp4", "video/avi", "video/mov"]
}


# 初始化模型
def init_model():
    global detector_model
    with model_lock:
        if detector_model is None:
            logger.info("正在加载模型...")
            try:
                detector_model = EnhancedSceneDetector(resource_manager, logger=logger)
                logger.info("模型加载完成")
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise


# 检查文件是否为允许的视频格式
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# 将帧号转换为时间戳字符串
def format_time(frame_number: int, fps: float) -> str:
    seconds = frame_number / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# 超时处理
def timeout_handler():
    raise TimeoutError("视频处理超时，请检查视频文件或调整超时时间设置")


# 验证请求数据
def validate_request_data(data):
    if not data:
        raise ValueError("请求体不能为空")

    # 验证必需参数
    required_fields = ["input_path", "output_path", "task_id"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"缺少必需参数: {', '.join(missing_fields)}")

    # 获取请求参数
    input_path = os.path.abspath(data["input_path"])
    output_path = os.path.abspath(data["output_path"])
    task_id = data["task_id"]
    threshold = float(data.get("threshold", 0.5))
    visualize = bool(data.get("visualize", False))
    video_split_audio_mode = data.get("video_split_audio_mode", AudioMode.UNMUTE)
    
    # 验证阈值范围
    if not 0 <= threshold <= 1:
        raise ValueError("阈值必须在0到1之间")

    # 验证任务ID格式
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("无效的任务ID")

    # 验证音频处理模式
    if video_split_audio_mode not in [AudioMode.MUTE, AudioMode.UNMUTE]:
        raise ValueError("不支持的音频处理模式")

    # 验证视频文件是否存在
    if not os.path.exists(input_path):
        raise ValueError("视频文件不存在")

    # 验证视频文件格式
    if not allowed_file(input_path):
        raise ValueError("不支持的视频文件格式")

    return (
        input_path,
        output_path,
        task_id,
        threshold,
        visualize,
        video_split_audio_mode,
    )


# 检测视频场景
def detect_video_scenes(input_path: str, threshold: float):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    try:
        # 直接使用全局模型实例
        logger.info("正在检测视频场景...")
        video_frames, single_frame_predictions, all_frame_predictions = (
            detector_model.predict_video(input_path)
        )
        scenes = detector_model.predictions_to_scenes(
            single_frame_predictions, threshold=threshold
        )

        return video_frames, scenes, single_frame_predictions, all_frame_predictions
    finally:
        cap.release()


# 写入视频片段
def write_video_segment(
    segment_clip,
    output_path,
    video_clip,
    video_split_audio_mode=AudioMode.UNMUTE,
    retries=3,
    delay=1,
):
    for attempt in range(retries):
        try:
            # 检查输出目录是否存在
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 获取原视频的编码参数
            original_video_bitrate = "8000k"
            original_audio_bitrate = "192k"
            original_audio_codec = "aac"
            
            # 检测是否支持NVENC编码器
            try:
                test_cap = cv2.VideoCapture()
                if hasattr(cv2, 'cudacodec') and cv2.cudacodec.getCudaEnabledDeviceCount() > 0:
                    codec = VIDEO_CODEC
                else:
                    codec = 'libx264'
                    logger.warning("NVENC编码器不可用，使用libx264作为备选编码器")
            except Exception:
                codec = 'libx264'
                logger.warning("检测编码器支持失败，使用libx264作为备选编码器")

            if video_clip.reader:
                if hasattr(video_clip.reader, "bitrate") and video_clip.reader.bitrate:
                    original_video_bitrate = str(int(video_clip.reader.bitrate)) + "k"
                if (
                    hasattr(video_clip.reader, "audio_bitrate")
                    and video_clip.reader.audio_bitrate
                ):
                    original_audio_bitrate = (
                        str(int(video_clip.reader.audio_bitrate)) + "k"
                    )
                if (
                    hasattr(video_clip.reader, "audio_codec")
                    and video_clip.reader.audio_codec
                ):
                    original_audio_codec = video_clip.reader.audio_codec

            cpu_count = os.cpu_count() or 4
            thread_count = max(1, cpu_count - 2)

            segment_clip.write_videofile(
                output_path,
                codec=VIDEO_CODEC,
                fps=video_clip.fps,
                bitrate=original_video_bitrate,
                preset="medium",
                threads=thread_count,
                audio=video_split_audio_mode
                == AudioMode.UNMUTE,  # 根据音频处理模式决定是否包含音频
                audio_codec=(
                    original_audio_codec
                    if video_split_audio_mode == AudioMode.UNMUTE
                    else None
                ),
                audio_bitrate=(
                    original_audio_bitrate
                    if video_split_audio_mode == AudioMode.UNMUTE
                    else None
                ),
                logger=None,
            )
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise


# 处理视频片段
def process_video_segments(
    video_clip, scenes, output_path, video_split_audio_mode=AudioMode.UNMUTE
):
    formatted_scenes = []
    video_duration = video_clip.duration

    for i, (start, end) in enumerate(scenes):
        try:
            start_time = start / video_clip.fps
            end_time = min(end / video_clip.fps, video_duration)

            # 如果起始时间已经超过视频总长度，跳过此片段
            if start_time >= video_duration:
                logger.warning(
                    f"场景 {i + 1} 的起始时间 {start_time}s 超出视频总长度 {video_duration}s，已跳过"
                )
                continue

            # 如果结束时间小于等于起始时间，跳过此片段
            if end_time <= start_time:
                logger.warning(
                    f"场景 {i + 1} 的时间区间无效 ({start_time}s - {end_time}s)，已跳过"
                )
                continue

            # 确保结束时间不超过视频总长度
            if end_time > video_duration:
                logger.warning(
                    f"场景 {i + 1} 的结束时间从 {end_time}s 调整为视频总长度 {video_duration}s"
                )
                end_time = video_duration

            segment_clip = video_clip.subclipped(start_time, end_time)

            # 为每个视频片段生成唯一文件名
            output_segment_path = os.path.join(output_path, f"segment_{i + 1}.mp4")
            write_video_segment(
                segment_clip, output_segment_path, video_clip, video_split_audio_mode
            )

            # 添加场景信息
            formatted_scenes.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(min(end, video_duration * video_clip.fps)),
                    "start_time": format_time(start, video_clip.fps),
                    "end_time": format_time(
                        int(end_time * video_clip.fps), video_clip.fps
                    ),
                    "output_path": output_segment_path,
                    "is_mute": video_split_audio_mode == AudioMode.MUTE,
                }
            )
        except Exception as e:
            logger.error(f"处理视频片段 {i + 1} 失败: {str(e)}")
            raise

    return formatted_scenes


# 处理视频场景分割请求 API
@app.route("/api/v1/scene-detection/process", methods=["POST"])
def process_scene_detection():
    try:
        # 解析和验证请求数据
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "无效的请求数据"}), 400
        try:
            # 验证请求数据
            (
                input_path,
                output_path,
                task_id,
                threshold,
                visualize,
                video_split_audio_mode,
            ) = validate_request_data(data)
        except ValueError as ve:
            return (
                jsonify({"status": "error", "message": str(ve), "task_id": task_id}),
                400,
            )

        logger.info(
            "开始处理视频场景分割",
            {
                "task_id": task_id,
                "input_path": input_path,
                "output_path": output_path,
                "threshold": threshold,
                "visualize": visualize,
            },
        )

        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)

        # 设置超时定时器
        timer = threading.Timer(SCENE_DETECTION_TIMEOUT, timeout_handler)
        timer.start()

        video_clip = None
        try:
            # 检测视频场景
            video_frames, scenes, single_frame_predictions, all_frame_predictions = (
                detect_video_scenes(input_path, threshold)
            )

            # 加载视频文件
            logger.info("正在切分场景...")
            try:
                video_clip = VideoFileClip(input_path)
                if not video_clip.reader or not hasattr(video_clip.reader, "fps"):
                    raise ValueError("无法正确加载视频文件，请检查视频格式是否正确")
            except Exception as e:
                logger.error(f"加载视频文件失败: {str(e)}")
                raise ValueError(f"加载视频文件失败: {str(e)}")

            # 处理视频片段
            formatted_scenes = process_video_segments(
                video_clip, scenes, output_path, video_split_audio_mode
            )

            # 如果需要可视化，生成预测结果的可视化图像
            if visualize:
                logger.info("正在生成预测可视化...")
                visualization = detector_model.visualize_predictions(
                    video_frames, [single_frame_predictions, all_frame_predictions]
                )
                visualization.save(f"{output_path}/predictions.png")

            logger.info(
                "处理完成",
                {
                    "task_id": task_id,
                    "scenes_count": len(scenes),
                    "output_dir": output_path,
                },
            )

            # 返回成功响应
            return jsonify(
                {
                    "status": "success",
                    "message": "处理完成",
                    "task_id": task_id,
                    "output_dir": output_path,
                    "data": formatted_scenes,
                }
            )

        finally:
            # 取消超时定时器
            timer.cancel()
            # 确保资源正确释放
            if video_clip is not None:
                video_clip.close()

    except TimeoutError as e:
        logger.error("处理超时", {"task_id": task_id, "error": str(e)})
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(e),
                    "task_id": task_id,
                    "output_dir": output_path,
                    "data": [],
                }
            ),
            408,
        )
    except ValueError as e:
        error_msg = str(e)
        logger.error("请求参数无效", {"error": error_msg})
        return (
            jsonify(
                {
                    "status": "error",
                    "message": error_msg,
                    "task_id": task_id,
                    "output_dir": output_path,
                    "data": [],
                }
            ),
            400,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error("处理过程中发生异常", {"task_id": task_id, "error": error_msg})
        return (
            jsonify(
                {
                    "status": "error",
                    "message": error_msg,
                    "task_id": task_id,
                    "output_dir": output_path,
                    "data": [],
                }
            ),
            500,
        )


# 添加全局错误处理
@app.errorhandler(Exception)
def handle_error(error):
    """处理所有未捕获的异常"""
    error_trace = traceback.format_exc()
    logger.error(f"未捕获的异常: {str(error)}\n{error_trace}")
    return (
        jsonify(
            {
                "status": "error",
                "message": "服务器内部错误",
                "error_type": type(error).__name__,
            }
        ),
        500,
    )


if __name__ == "__main__":
    try:
        # 服务启动时初始化模型
        init_model()
        app.run(host="0.0.0.0", port=9000)
    except Exception as e:
        logger.error(f"视频场景切割服务启动失败: {str(e)}")
        sys.exit(1)
