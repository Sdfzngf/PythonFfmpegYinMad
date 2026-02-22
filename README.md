# PythonFfmpegYinMad

使用FFmpeg的全自动音MAD视频制作脚本

帮助信息：

```
usage: mel.py [-h] [--midi MIDI] [--src SRC] [--out OUT] [--temp-dir TEMP_DIR]
              [--mix-chunk MIX_CHUNK] [--resume] [--resume-from RESUME_FROM]
              [--crf CRF] [--preset PRESET]

根据 MIDI 与基准素材生成音 MAD 视频。 本脚本由 Copilot 编写

options:
  -h, --help            show this help message and exit
  --midi MIDI           输入 MIDI 文件
  --src SRC             中央 C(60) 的基准视频
  --out OUT             输出视频
  --temp-dir TEMP_DIR   临时文件相对路径
  --mix-chunk MIX_CHUNK
                        每次 ffmpeg 合成的音符片段数量
  --resume              从现有临时文件恢复
  --resume-from RESUME_FROM
                        从指定音符索引恢复（从 0 开始）
  --crf CRF             全局 x264 CRF 值
  --preset PRESET       全局 x264 preset
```
