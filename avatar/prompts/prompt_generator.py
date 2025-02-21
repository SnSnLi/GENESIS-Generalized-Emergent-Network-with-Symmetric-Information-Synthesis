import os
import os.path as osp
from typing import Dict, Any, List
from avatar.models.sden_model import (
    final_features,
    emerged_features,
    topo_features,
    entropy_ranking
)


class PromptGenerator:
    def __init__(self, discovery_output: Dict[str, Any]):
        """
        Initialize the prompt generator with discovery output
        
        Args:
            discovery_output: Output from discovery module containing causal relationships
        """
        self.discovery_output = discovery_output
        self.template_backup = {}

    def _load_template(self, template_name: str) -> str:
        """加载模板文件并缓存"""
        template_path = osp.join(self.template_dir, f"avatar_{template_name}.txt")
        
        try:
            if template_path not in self.template_cache:
                with open(template_path, 'r') as f:
                    self.template_cache[template_path] = f.read()
            return self.template_cache[template_path]
        except FileNotFoundError:
            self.logger.error(f"模板文件未找到: {template_path}")
            raise
        except IOError as e:
            self.logger.error(f"读取模板失败: {str(e)}")
            raise

    def _backup_template(self, template_name: str) -> None:
        """备份原始模板内容"""
        template_path = osp.join(self.template_dir, f"avatar_{template_name}.txt")
        try:
            with open(template_path, 'r') as f:
                self.template_backup[template_name] = f.read()
        except IOError as e:
            self.logger.error(f"备份模板失败: {str(e)}")

    def _restore_template(self, template_name: str) -> None:
        """恢复原始模板内容"""
        template_path = osp.join(self.template_dir, f"avatar_{template_name}.txt")
        if template_name in self.template_backup:
            try:
                with open(template_path, 'w') as f:
                    f.write(self.template_backup[template_name])
            except IOError as e:
                self.logger.error(f"恢复模板失败: {str(e)}")

    def generate_prompt(self, 
                       prompt_type: str,
                       final_features: str,
                       emerged_raw: str,
                       entropy_weights: str) -> str:
        """
        生成指定类型的提示模板
        
        Args:
            prompt_type: 提示类型 (comparator/improver/initializer)
            final_features: 最终特征描述
            emerged_raw: 原始涌现特征
            entropy_weights: 熵权重分布
            
        Returns:
            格式化后的提示内容
        """
        # 验证必要参数
        if not all([final_features, emerged_raw, entropy_weights]):
            raise ValueError("缺少必要参数")
            
        # 加载基础模板
        template = self._load_template(prompt_type)
        
        # 替换占位符
        replacements = {
            '{final_features}': final_features,
            '{emerged_raw}': emerged_raw,
            '{entropy_weights}': entropy_weights
        }
        
        prompt = template
        for ph, value in replacements.items():
            prompt = prompt.replace(ph, value)
            
        # 验证模板完整性
        if not self._validate_template(prompt):
            self.logger.warning("生成模板可能不完整，缺少必要占位符")
            
        return prompt

    def _validate_template(self, prompt: str) -> bool:
        """验证模板完整性"""
        required_placeholders = {'{final_features}', '{emerged_raw}'}
        present_placeholders = set(re.findall(r'{\w+}', prompt))
        return required_placeholders.issubset(present_placeholders)


    def generate_comparator_prompt(
        final_features: str,
        emerged_raw: str,
        entropy_weights: str
    ) -> str:
        
        with open('root/onethingai-tmp/avatar/avatar/prompts/avatar_comparator.txt', 'r') as f:
            template = f.read()
        
        # 替换占位符
        prompt = template.replace('{final_features}', final_features)\
                        .replace('{emerged_raw}', emerged_raw)\
                        .replace('{entropy_weights}', entropy_weights)
                       
        
        with open('root/onethingai-tmp/avatar/avatar/prompts/avatar_comparator.txt', 'w') as f:
            f.write(prompt)
        
        return prompt

    def generate_avatar_initialize_prompt(final_features, topo_features):

        with open('root/onethingai-tmp/avatar/avatar/prompts/avatar_initialize_actions_flickr30k_ent.txt', 'r') as f:
            template = f.read()

        prompt = template.replace('{final_features}', final_features)\
                        .replace('{topo_features}', topo_features)\
                      
                        
        with open('avatar_initialize_actions_flickr30k_ent.txt', 'w') as f:
            f.write(prompt)
        
        return prompt

    
    def _get_prompt(self, name: str = 'initialize_actions', **kwargs: Any) -> str:
        
        prompt_path = {
            
            'initialize_actions': 'avatar/prompts/avatar_initialize_actions_flickr30k_ent.txt',
            'improve_actions': 'avatar/prompts/avatar_improve_actions.txt',
            'comparator': 'avatar/prompts/avatar_comparator.txt',
            'assign_group': 'avatar/rompts/preprocess_group_assignment.txt'
            
        }
        current_dir = osp.dirname(osp.abspath(__file__))
        prompt_path = {key: osp.join(current_dir, '..', path) \
            for key, path in prompt_path.items()}

        if name == 'comparator':
            prompt = read_from_file(prompt_path[name])
            prompt = template.replace('{final_features}', final_features)\
                            .replace('{emerged_raw}', emerged_raw)\
                            .replace('{entropy_weights}', entropy_weights)


            

        elif name == 'initialize_actions':
            
            
                prompt = read_from_file(prompt_path[name])
                sample_indices = kwargs['sample_indices']
                qa_dataset = kwargs['qa_dataset']
                pattern = kwargs['pattern']
                func_call_description = '\n'.join(['- ' + func.func_format + '. ' + func.description for func in self.funcs])
                example_queries = '\n'.join([f'Q{i+1}: ' + qa_dataset[idx][0] for i, idx in enumerate(sample_indices)])
                prompt = template.replace('{final_features}', final_features)\
                                .replace('{topo_features}', topo_features)\
                

        elif name == 'improve_actions':
            prompt = read_from_file(prompt_path[name])
            debug_message = kwargs['debug_message']
            feedback_message = kwargs['feedback_message']
            query = kwargs['query']
            candidate_ids = kwargs['candidate_ids']
            debug_message = debug_message.strip(' \n')
            prompt = prompt.replace('<debug_message>', '"\n' + debug_message + '\n"' if len(debug_message) else 'No output')
            prompt = prompt.replace('<feedback_message>', feedback_message)
            prompt = prompt.replace('<input_query>', '"' + query + '"' if len(query) else 'Not specified')
            prompt = prompt.replace('<size_of_candidate_ids>', str(len(candidate_ids)) if len(candidate_ids) else 'Not specified')

        elif name == 'assign_group':
            prompt = read_from_file(prompt_path[name])
            query = kwargs['query']
            group_patterns = kwargs['group_patterns']
            prompt = prompt.replace('<query>', query)
            prompt = prompt.replace('<group_patterns>', group_patterns)

        return prompt


        

