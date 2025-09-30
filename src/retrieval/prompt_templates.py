"""
Prompt templates for RAG system
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata"""
    template: str
    description: str
    required_variables: List[str]
    optional_variables: List[str] = None

class PromptTemplates:
    """
    Collection of optimized prompt templates for RAG systems
    """
    
    @staticmethod
    def get_basic_rag_prompt() -> PromptTemplate:
        """Basic RAG prompt template"""
        template = """Based on the following context, provide a clear and accurate answer to the user's question.

Context:
{context}

Question: {query}

Instructions:
- Use only information from the provided context
- Be specific and cite relevant sources when possible
- If the answer isn't available in the context, say so clearly
- Provide a well-structured response

Answer:"""
        
        return PromptTemplate(
            template=template,
            description="Basic RAG prompt for standard Q&A",
            required_variables=["context", "query"]
        )
    
    @staticmethod
    def get_advanced_synthesis_prompt() -> PromptTemplate:
        """Advanced synthesis prompt for complex analysis"""
        template = """You are an expert analyst with access to multiple information sources. Your task is to provide a comprehensive, well-reasoned answer based on the provided context.

{context}

QUERY: {query}

INSTRUCTIONS:
1. Analyze all the provided sources carefully
2. Identify key themes, patterns, and relationships across sources
3. Synthesize information from multiple sources where relevant
4. Highlight any conflicting information and explain potential reasons
5. Provide a structured, comprehensive answer
6. Cite specific sources for key claims
7. If information is insufficient, clearly state what additional information would be helpful

RESPONSE FORMAT:
- Start with a direct answer to the query
- Provide supporting evidence from the sources
- Include analysis of relationships between different pieces of information
- End with any important caveats or limitations

Answer:"""
        
        return PromptTemplate(
            template=template,
            description="Advanced synthesis prompt for complex analysis and multi-source reasoning",
            required_variables=["context", "query"]
        )
    
    @staticmethod
    def get_conversational_prompt() -> PromptTemplate:
        """Conversational RAG prompt with chat history"""
        template = """You are a helpful assistant engaging in a conversation. Use the provided context to answer the current question while being aware of the conversation history.

Context:
{context}

Conversation History:
{chat_history}

Current Question: {query}

Instructions:
- Consider the conversation history for context
- Use information from the provided context to answer
- Maintain a conversational and helpful tone
- If the question refers to previous parts of the conversation, acknowledge that
- Provide clear, accurate information based on the context

Answer:"""
        
        return PromptTemplate(
            template=template,
            description="Conversational prompt that considers chat history",
            required_variables=["context", "query"],
            optional_variables=["chat_history"]
        )
    
    @staticmethod
    def get_technical_prompt() -> PromptTemplate:
        """Technical documentation prompt"""
        template = """You are a technical documentation assistant. Based on the provided technical context, answer the user's question with precision and clarity.

Technical Context:
{context}

Question: {query}

Instructions:
- Provide technically accurate information
- Include relevant code examples, configuration details, or technical specifications from the context
- Use appropriate technical terminology
- Structure your answer with clear sections if needed
- If the question involves troubleshooting, provide step-by-step guidance
- Cite specific sections or sources when referencing technical details

Technical Answer:"""
        
        return PromptTemplate(
            template=template,
            description="Technical documentation prompt for code, configuration, and technical Q&A",
            required_variables=["context", "query"]
        )
    
    @staticmethod
    def get_comparative_prompt() -> PromptTemplate:
        """Comparative analysis prompt"""
        template = """You are tasked with providing a comparative analysis based on the provided information sources.

Information Sources:
{context}

Comparison Query: {query}

Instructions:
- Compare and contrast the relevant information from different sources
- Identify similarities and differences
- Highlight advantages and disadvantages where applicable
- Provide a balanced perspective
- Use specific examples from the sources
- Organize your response with clear comparisons
- Conclude with a summary of key insights

Comparative Analysis:"""
        
        return PromptTemplate(
            template=template,
            description="Comparative analysis prompt for comparing multiple sources or concepts",
            required_variables=["context", "query"]
        )
    
    @staticmethod
    def get_summarization_prompt() -> PromptTemplate:
        """Document summarization prompt"""
        template = """Summarize the following content in response to the user's specific request.

Content to Summarize:
{context}

Summarization Request: {query}

Instructions:
- Focus on the aspects most relevant to the user's request
- Provide a concise yet comprehensive summary
- Maintain the key information and important details
- Structure the summary logically
- Highlight the most important points
- If the user asks for a specific type of summary (executive summary, technical overview, etc.), adapt accordingly

Summary:"""
        
        return PromptTemplate(
            template=template,
            description="Document summarization prompt for various summarization needs",
            required_variables=["context", "query"]
        )
    
    @staticmethod
    def get_creative_prompt() -> PromptTemplate:
        """Creative writing prompt based on context"""
        template = """Using the provided context as inspiration and factual grounding, respond to the creative request.

Contextual Information:
{context}

Creative Request: {query}

Instructions:
- Use the context as a factual foundation
- Be creative while staying grounded in the provided information
- Maintain accuracy for any factual claims
- Adapt your creative style to match the request
- If writing fiction, clearly distinguish between factual context and creative elements

Creative Response:"""
        
        return PromptTemplate(
            template=template,
            description="Creative writing prompt that uses context as inspiration",
            required_variables=["context", "query"]
        )
    
    @staticmethod
    def get_template_by_type(template_type: str) -> PromptTemplate:
        """Get template by type name"""
        template_map = {
            'basic': PromptTemplates.get_basic_rag_prompt,
            'advanced': PromptTemplates.get_advanced_synthesis_prompt,
            'conversational': PromptTemplates.get_conversational_prompt,
            'technical': PromptTemplates.get_technical_prompt,
            'comparative': PromptTemplates.get_comparative_prompt,
            'summary': PromptTemplates.get_summarization_prompt,
            'creative': PromptTemplates.get_creative_prompt
        }
        
        if template_type not in template_map:
            # Default to basic if type not found
            template_type = 'basic'
        
        return template_map[template_type]()
    
    @staticmethod
    def format_prompt(
        template: PromptTemplate, 
        variables: Dict[str, Any]
    ) -> str:
        """Format a prompt template with provided variables"""
        
        # Check required variables
        missing_vars = [var for var in template.required_variables if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # Set default values for optional variables
        format_vars = variables.copy()
        if template.optional_variables:
            for opt_var in template.optional_variables:
                if opt_var not in format_vars:
                    format_vars[opt_var] = ""
        
        try:
            return template.template.format(**format_vars)
        except KeyError as e:
            raise ValueError(f"Template variable not provided: {e}")
    
    @staticmethod
    def get_available_templates() -> Dict[str, str]:
        """Get list of available template types with descriptions"""
        return {
            'basic': 'Basic RAG prompt for standard Q&A',
            'advanced': 'Advanced synthesis prompt for complex analysis',
            'conversational': 'Conversational prompt with chat history',
            'technical': 'Technical documentation prompt',
            'comparative': 'Comparative analysis prompt',
            'summary': 'Document summarization prompt',
            'creative': 'Creative writing prompt'
        }