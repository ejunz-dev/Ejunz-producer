import textwrap

from app.mcq_generation import MCQGenerator

def show_result(generated: str, answer: str, context:str, original_question: str = ''):
    print('Context:')
    for wrap in textwrap.wrap(context, width=120):
        print(wrap)
    print()
    print('Question:')
    print(generated)
    print('Answer:')
    print(answer)
    print('-----------------------------')


MCQ_Generator = MCQGenerator(True)

# 五行相关的上下文
context_wuxing = '''The Five Elements, also known as Wu Xing, are a Chinese philosophical concept that describes the interaction and relationship between five fundamental elements: Wood (木), Fire (火), Earth (土), Metal (金), and Water (水). These elements are not substances in a literal sense but rather categories that represent different types of energy or dynamic processes in the universe.

- Wood symbolizes growth, vitality, and creativity.
- Fire represents energy, passion, and transformation.
- Earth is associated with stability, nurturing, and balance.
- Metal signifies structure, strength, and discipline.
- Water is linked to adaptability, reflection, and wisdom.

The elements interact in two primary cycles: the Generating Cycle (相生) and the Overcoming Cycle (相克). In the Generating Cycle, each element supports the next: Wood feeds Fire, Fire produces Earth (ash), Earth bears Metal, Metal enriches Water (condensation), and Water nourishes Wood. In the Overcoming Cycle, each element controls another: Wood overcomes Earth, Earth absorbs Water, Water extinguishes Fire, Fire melts Metal, and Metal chops Wood.

The Five Elements theory is deeply rooted in Chinese culture and has applications in traditional medicine, feng shui, astrology, martial arts, and philosophy.'''

# 使用五行的上下文生成问题
MCQ_Generator.generate_mcq_questions(context_wuxing, 10)
