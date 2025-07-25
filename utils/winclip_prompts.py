state_level_normal_prompts = [
    "{classname}",
    "flawless {classname}",
    "perfect {classname}",
    "unblemished {classname}",
    "{classname} without flaw",
    "{classname} without defect",
    "{classname} without damage",
]

state_level_abnormal_prompts = [
    "{size} damaged {{classname}} {position}",
    "{{classname}} with {size} flaw {position}",
    "{{classname}} with {size} defect {position}",
    "{{classname}} with {size} damage {position}",
]

template_level_prompts = [
    "a cropped photo of the {}",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
    "a close-up photo of the {}",
    "a bright photo of a {}",
    "a bright photo of the {}",
    "a dark photo of the {}",
    "a dark photo of a {}",
    "a jpeg corrupted photo of a {}",
    "a jpeg corrupted photo of the {}",
    "a blurry photo of the {}",
    "a blurry photo of a {}",
    "a photo of a {}",
    "a photo of the {}",
    "a photo of a small {}",
    "a photo of the small {}",
    "a photo of a large {}",
    "a photo of the large {}",
    "a photo of the {} for visual inspection",
    "a photo of a {} for visual inspection",
    "a photo of the {} for anomaly detection",
    "a photo of a {} for anomaly detection",
]

anomaly_size_prompts = [
    "",
    "tiny",
    "small",
    "big",
    "large",
    "huge",
    "light",
    "severe",
]

anomaly_position_prompts = [
    "",
    "in the middle",
    "in the center",
    "on the left",
    "on the right",
    "on the top",
    "on the bottom",
    "on the top left",
    "on the bottom left",
    "on the top right",
    "on the bottom right",
    "everywhere",
]

# 定义了一个名为 generate_prompts 的函数，用于生成异常和正常状态下的提示语句
def generate_prompts(anomaly_size=False, anomaly_position=False):
    prompts = {
        "abnormal": {
            "instruction": "",
            "prompts": [],
        },
        "normal": {
            "instruction": "",
            "prompts": [],
        },
    }

    size_prompts = anomaly_size_prompts if anomaly_size else [""]
    position_prompts = anomaly_position_prompts if anomaly_position else [""]
    # 生成异常提示：遍历模板级别提示、状态级别异常提示、大小提示和位置提示，组合生成异常提示并添加到 prompts["abnormal"]["prompts"] 列表中
    for template_prompt in template_level_prompts:
        for state_prompt in state_level_abnormal_prompts:
            for size_prompt in size_prompts:
                for position_prompt in position_prompts:
                    prompts["abnormal"]["prompts"].append(
                        " ".join(
                            template_prompt.format(state_prompt)
                            .format(size=size_prompt, position=position_prompt)
                            .split()
                        )
                    )
    # 生成正常提示：遍历模板级别提示和状态级别正常提示，组合生成正常提示并添加到 prompts["normal"]["prompts"] 列表中
    for template_prompt in template_level_prompts:
        for state_prompt in state_level_normal_prompts:
            prompts["normal"]["prompts"].append(template_prompt.format(state_prompt))

    return prompts
