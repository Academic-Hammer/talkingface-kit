1、AED：输入参数2个，generated_folder：生成的图片文件夹；driving_folder：输入的图片文件夹，就是data/0/imgs
指令示例 python AED.py --generated_folder "Jae-in" --driving_folder "base_model"

2、APD：输入参数2个，generated_folder：生成的图片文件夹；driving_folder：输入的图片文件夹，就是data/0/imgs
指令示例 python APD.py --generated_folder "Jae-in" --driving_folder "base_model"

3、ID loss：输入参数2个，generated_folder：生成的图片文件夹；input_image_path：输入的源图片路径，就是src_0.png
指令示例 python ID_loss.py --input_image_path "src_0.png" --generated_folder "Jae-in"

4、LPIPS：输入参数2个，generated_folder：生成的图片文件夹；driving_folder：输入的图片文件夹，就是data/0/imgs
指令示例 python LPIPS.py --driving_folder "base_model" --generated_folder "Jae-in"

5、LSE-C：输入参数1个，generated_folder：生成的图片文件夹
指令示例 python LSE-C.py --generated_folder "Jae-in"

6、LSE-D：输入参数2个，generated_folder：生成的图片文件夹；driving_folder：输入的图片文件夹，就是data/0/imgs
指令示例 python LSE-D.py --generated_folder "Jae-in" --driving_folder "base_model"

7、SSIM：输入参数2个，generated_folder：生成的图片文件夹；driving_folder：输入的图片文件夹，就是data/0/imgs
指令示例 python SSIM.py --generated_folder "Jae-in" --driving_folder "base_model"

8、FID：输入参数2个，分别是生成的图片文件夹；输入的图片文件夹，就是data/0/imgs
指令示例 python -m pytorch_fid "base_model" "Jae-in"
其中"base_model"是输入的图片文件夹，"Jae-in"是生成的图片文件夹
这个评估指标比较特殊，没有对应文件，直接在终端输入即可

9、NIQE(位于niqe-master/niqe.py)：输入参数1个，generated_folder：生成的图片文件夹
指令示例 python niqe-master/niqe.py --generated_folder "Jae-in"

以上所有指标的文件最终都是在终端输出结果，结果是一个表示指标的数字