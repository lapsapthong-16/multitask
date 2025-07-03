import json

def flatten_reddit_data(json_file_path, output_json_path):
    """
    Flatten Reddit JSON data into a new JSON with posts and comments as separate entries.
    
    Args:
        json_file_path: Path to the input JSON file
        output_json_path: Path for the output JSON file
    """
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flattened_rows = []
    
    # Process each post in the JSON
    for post in data:
        # Create a row for the post itself
        post_title = post.get('title', '')
        post_selftext = post.get('selftext', '')
        
        # Combine title and selftext with a space separator, handle empty selftext
        if post_selftext:
            text_content = f"{post_title} {post_selftext}".strip()
        else:
            text_content = post_title.strip()
        
        post_row = {
            'id': post.get('id', ''),
            'text_content': text_content,
            'type': 'post',
            'score': post.get('score', 0),
            'subjectivity': post.get('subjectivity', 0.0),
            'include_for_labeling': ''
        }
        flattened_rows.append(post_row)
        
        # Process all comments for this post
        comments = post.get('extracted_comments', [])
        for comment in comments:
            comment_row = {
                'id': comment.get('comment_id', ''),
                'text_content': comment.get('comment_text', ''),
                'type': 'comment',
                'score': comment.get('comment_score', 0),
                'subjectivity': comment.get('comment_subjectivity', 0.0),
                'include_for_labeling': ''
            }
            flattened_rows.append(comment_row)
    
    # Export to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(flattened_rows, f, ensure_ascii=False, indent=2)
    
    # Print summary statistics
    total_rows = len(flattened_rows)
    posts = [row for row in flattened_rows if row['type'] == 'post']
    comments = [row for row in flattened_rows if row['type'] == 'comment']
    
    print(f"✅ Data successfully flattened and exported to: {output_json_path}")
    print(f"\n📊 Summary:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Posts: {len(posts):,}")
    print(f"   Comments: {len(comments):,}")
    
    if posts:
        post_scores = [p['score'] for p in posts]
        post_subjectivities = [p['subjectivity'] for p in posts]
        print(f"\n📈 Post statistics:")
        print(f"   Scores - Min: {min(post_scores)}, Max: {max(post_scores)}, Avg: {sum(post_scores)/len(post_scores):.1f}")
        print(f"   Subjectivity - Min: {min(post_subjectivities):.3f}, Max: {max(post_subjectivities):.3f}, Avg: {sum(post_subjectivities)/len(post_subjectivities):.3f}")
    
    if comments:
        comment_scores = [c['score'] for c in comments]
        comment_subjectivities = [c['subjectivity'] for c in comments]
        print(f"\n💬 Comment statistics:")
        print(f"   Scores - Min: {min(comment_scores)}, Max: {max(comment_scores)}, Avg: {sum(comment_scores)/len(comment_scores):.1f}")
        print(f"   Subjectivity - Min: {min(comment_subjectivities):.3f}, Max: {max(comment_subjectivities):.3f}, Avg: {sum(comment_subjectivities)/len(comment_subjectivities):.3f}")
    
    return flattened_rows

def preview_data(json_file_path, num_rows=5):
    """
    Preview the flattened JSON data.
    
    Args:
        json_file_path: Path to the JSON file
        num_rows: Number of rows to preview
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n👀 Preview of first {num_rows} entries:")
    for i, entry in enumerate(data[:num_rows]):
        print(f"\n{i+1}. ID: {entry['id']}")
        print(f"   Type: {entry['type']}")
        print(f"   Score: {entry['score']}")
        print(f"   Subjectivity: {entry['subjectivity']}")
        print(f"   Text: {entry['text_content'][:100]}{'...' if len(entry['text_content']) > 100 else ''}")

if __name__ == "__main__":
    # File paths - update these to match your files
    json_file_path = "reddit-posts-5.json"
    output_json_path = "flat5.json"
    
    # Flatten the data
    flattened_data = flatten_reddit_data(json_file_path, output_json_path)
    
    # Preview the results
    preview_data(output_json_path, num_rows=5)
    
    print(f"\n✨ Next steps:")
    print(f"   1. Open {output_json_path} in your preferred JSON editor")
    print(f"   2. Fill in the 'include_for_labeling' field with your labels")
    print(f"   3. Use this data for training your model")