"""Local-only recording interface. No model weights are downloaded here."""
import argparse
from pathlib import Path

import gradio as gr

from common import choose_device, load_checkpoint
from predict import predict_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/experiment01/best.pt'))
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)

    def classify(image):
        if image is None:
            raise gr.Error('请先上传只含一个完整纸盒的裁剪图片。')
        return predict_image(model, image, device)

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Image(type='pil', label='上传单个纸盒的裁剪图', sources=['upload']),
        outputs=gr.Label(num_top_classes=6, label='类别 0–5 与模型分数'),
        title='纸盒六分类 · PyTorch 迁移学习教学示例',
        description=('基于 PyTorch 官方教程改编的 ResNet18 分类器。上传只含一个完整纸盒的裁剪图，'
                     '输出六类模型分数。整体准确率需在测试集统计，机器人抓取还需要三维定位与控制。'),
        article=f'当前权重：{args.checkpoint.name}，选中轮次：{checkpoint["epoch"]}。类别对应关系见 reference_labels。',
        flagging_mode='never', analytics_enabled=False,
    )
    demo.launch(server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=True)


if __name__ == '__main__':
    main()
