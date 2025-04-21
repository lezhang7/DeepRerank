import re

from mathruler.grader import extract_boxed_content, grade_answer


def response_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if grade_answer(given_answer, ground_truth.strip()):
            return 1.0

    except Exception:
        pass

    return 0.0

def ranking_format_reward(predict_str: str) -> float:
    """
    Checks if the content within <answer> tags follows the ranking format [number] > [number] > [number]
    Spaces are ignored. Returns 1.0 if the format is correct, 0.0 if incorrect.
    """
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        if content_match:
            answer_content = content_match.group(1).strip()
            # Match format [number] > [number] > ... ignoring spaces
            ranking_pattern = re.compile(r"^\s*\[\s*\d+\s*\]\s*>\s*\[\s*\d+\s*\](\s*>\s*\[\s*\d+\s*\])*\s*$")
            if ranking_pattern.match(answer_content):
                return 1.0
    except Exception:
        pass
    
    return 0.0

def compute_score(predict_str: str, ground_truth: str) -> float:
    return 0.5 * accuracy_reward(predict_str, ground_truth) + 0.25 * response_format_reward(predict_str) + 0.25 * ranking_format_reward(predict_str) 


if __name__ == "__main__":
    predict_str = "<think>I think the answer is 1</think> <answer> [1] > [2] > [3]    </answer>"
    print(f"response_format_reward: {response_format_reward(predict_str)}")
    print(f"ranking_format_reward: {ranking_format_reward(predict_str)}")
    print(f"accuracy_reward: {accuracy_reward(predict_str, '1')}")
    print(f"compute_score: {compute_score(predict_str, '1')}")
