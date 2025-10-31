
### 说明：
1. **项目介绍**：主要是增加了infinitetalk对于多图的支持。
2. **使用方法**：使用下面的示例工作流即可。
3. **示例视频**：[示例视频](https://github.com/starsFriday/ComfyUI-WanVideoWrapper-Enhanced-Infinitetalk/blob/main/Wan_00040-audio.mp4)
4. **示例工作流和节点**：<img width="879" height="1122" alt="db712424c4b427f076ee1f1287489312" src="https://ai.static.ad2.cc/workflow-multi_infinitetalkv2.png" />

<img width="540" height="959" alt="db712424c4b427f076ee1f1287489312" src="https://ai.static.ad2.cc/node.png" />

5.**节点说明**：
修改```image_count```参数对应输入图片的数量,每一张图只在当前的```frame_window_size```窗口帧内生效，从```start_image```开始依次往后切换，当图片数量不够时则一直使用最后一张图片生成
