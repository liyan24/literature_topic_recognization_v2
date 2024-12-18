def format_time(seconds):
    """将秒数转换为"时分秒"格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60, 1)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0:
        parts.append(f"{minutes}分钟")
    if seconds > 0 or not parts:  # 如果没有小时和分钟，则显示秒
        parts.append(f"{seconds}秒")
    
    return "".join(parts)

def estimate_remaining_time(elapsed_time, current, total):
    """估算剩余时间"""
    if current == 0:
        return "计算中..."
    
    avg_time_per_record = elapsed_time / current
    remaining_records = total - current
    estimated_remaining_time = avg_time_per_record * remaining_records
    
    return format_time(estimated_remaining_time)