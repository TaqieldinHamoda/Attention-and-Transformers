from manim import *
import numpy as np

class Attention(Scene):
	def softmax(self, xs):
		return np.exp(xs)/np.sum(np.exp(xs))


	def context_vector(self, hidden_states, s_vector):
		annotations = np.dot(hidden_states.transpose(), s_vector)
		soft_annotations = self.softmax(annotations)
		context_vector = soft_annotations[0]*hidden_states[:, 0] + soft_annotations[1]*hidden_states[:, 1] + soft_annotations[2]*hidden_states[:, 2]

		return (annotations.round(2), soft_annotations.round(2), context_vector.round(2))


	def construct(self):
		# Math Portion
		h_states = np.array([
			[1, 0, 3],
			[1, 5, -1],
			[9, 6, 2],
		])

		# TODO:Fix up s_1 and s_2 so the numbers match up with results
		s_vectors = np.array([
			[0, 2, 5],
			[1, 3, 6],
			[1, 4, 7],
		])

		anno1, s_anno1, con1 = self.context_vector(h_states, s_vectors[:, 0])
		anno2, s_anno2, con2 = self.context_vector(h_states, s_vectors[:, 1])

		# Displayed Text Portion
		h_text =[
			 DecimalMatrix(
				[[h_states[0, 0]], [h_states[1, 0]], [h_states[2, 0]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
			 DecimalMatrix(
				[[h_states[0, 1]], [h_states[1, 1]], [h_states[2, 1]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
			 DecimalMatrix(
				[[h_states[0, 2]], [h_states[1, 2]], [h_states[2, 2]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
		]

		hm_text = DecimalMatrix(
			h_states.transpose(),
			element_to_mobject_config={"num_decimal_places": 0},
			left_bracket="[",
			right_bracket="]"
		)

		s_text =[
			 DecimalMatrix(
				[[s_vectors[0, 0]], [s_vectors[1, 0]], [s_vectors[2, 0]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
			 DecimalMatrix(
				[[s_vectors[0, 1]], [s_vectors[1, 1]], [s_vectors[2, 1]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
			 DecimalMatrix(
				[[s_vectors[0, 2]], [s_vectors[1, 2]], [s_vectors[2, 2]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
		]

		# Deep copy of s_text
		s_math = [
			 DecimalMatrix(
				[[s_vectors[0, 0]], [s_vectors[1, 0]], [s_vectors[2, 0]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
			 DecimalMatrix(
				[[s_vectors[0, 1]], [s_vectors[1, 1]], [s_vectors[2, 1]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
			 DecimalMatrix(
				[[s_vectors[0, 2]], [s_vectors[1, 2]], [s_vectors[2, 2]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
		]

		h_labels = [
			MathTex(r"h^{(0)}"),
			MathTex(r"h^{(1)} ="),
			MathTex(r"h^{(2)} ="),
			MathTex(r"h^{(3)} ="),
		]

		s_labels = [
			MathTex(r"s^{(0)} ="),
			MathTex(r"s^{(1)} ="),
			MathTex(r"s^{(2)} ="),
		]

		hs_labels = [
			Matrix(
				[[r"s^{(0)} \cdot h^{(1)}"], [r"s^{(0)} \cdot h^{(2)}"], [r"s^{(0)} \cdot h^{(3)}"]],
            			element_alignment_corner=UL,
            			left_bracket="[",
		            	right_bracket="]"
			),
			Matrix(
				[[r"s^{(1)} \cdot h^{(1)}"], [r"s^{(1)} \cdot h^{(2)}"], [r"s^{(1)} \cdot h^{(3)}"]],
            			element_alignment_corner=UL,
            			left_bracket="[",
		            	right_bracket="]"
			),
		]

		anno_label = [MathTex(r"\alpha_{1} ="), MathTex(r"\alpha_{2} =")]
		anno_text = [
			 DecimalMatrix(
				[[anno1[0]], [anno1[1]], [anno1[2]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
			 DecimalMatrix(
				[[anno2[0]], [anno2[1]], [anno2[2]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]"
			),
		]

		s_anno_text = [
			 DecimalMatrix(
				[[s_anno1[0]], [s_anno1[1]], [s_anno1[2]]],
				element_to_mobject_config={"num_decimal_places": 2},
				left_bracket="[",
				right_bracket="]"
			),
			 DecimalMatrix(
				[[s_anno2[0]], [s_anno2[1]], [s_anno2[2]]],
				element_to_mobject_config={"num_decimal_places": 2},
				left_bracket="[",
				right_bracket="]"
			),
		]

		# Get it, con_text vector
		con_label = MathTex(r"c = \alpha_{1}h^{(1)} + \alpha_{2}h^{(2)} + \alpha_{3}h^{(3)}")
		con_text = [
			MathTex(r"c_{1} = [1]h^{(1)} + [2]h^{(2)} + [3]h^{(3)}".replace("[1]", f"{s_anno1[0]}").replace("[2]", f"{s_anno1[1]}").replace("[3]", f"{s_anno1[2]}")),
			MathTex(r"c_{2} = [1]h^{(1)} + [2]h^{(2)} + [3]h^{(3)}".replace("[1]", f"{s_anno2[0]}").replace("[2]", f"{s_anno2[1]}").replace("[3]", f"{s_anno2[2]}")),
		]

		c_label = [MathTex(r"c_{1} ="), MathTex(r"c_{2} =")]
		c_text = [
			DecimalMatrix(
				[[con1[0]], [con1[1]], [con1[2]]],
				element_to_mobject_config={"num_decimal_places": 2},
				left_bracket="[",
				right_bracket="]"
			),
			DecimalMatrix(
				[[con1[0]], [con1[1]], [con1[2]]],
				element_to_mobject_config={"num_decimal_places": 2},
				left_bracket="[",
				right_bracket="]"
			),
		]

		input_labels = [
			MathTex("x^{(1)}"),
			MathTex("x^{(2)}"),
			MathTex("x^{(3)}"),
			MathTex("<EOS>"),
		]

		softmax_label = [MathTex(r"softmax("), MathTex(r")")]
		softmax_func = MathTex(r"softmax = \frac{e^{z_{i}}}{\sum^K_{j=1} e^{z_{j}}}")

		english = [
			Text("I", color=GREEN),
			Text("Eat", color=GREEN),
			Text("Apples", color=GREEN),
		]

		spanish = [
			Text("Como", color=RED),
			Text("Manzanas", color=RED),
		]


		# Creating Scene
		title = Text("Attention in RNNs", gradient=(BLUE, GREEN)).scale(1.5)
		self.play(Write(title))
		self.wait(2)
		self.play(FadeOut(title))

		# Encoder Creation
		self.wait(1)
		self.play(Write(h_labels[0].move_to(2*DOWN + 5.75*LEFT)))

		self.wait(1)
		self.play(Create(Arrow(start=LEFT, end=RIGHT).scale(0.5).next_to(h_labels[0], RIGHT)))
		self.play(Write(h_labels[1].move_to(2*DOWN + 3.5*LEFT)))
		self.play(Write(english[0].scale(0.7).next_to(h_labels[1], 4*DOWN)))
		self.play(Create(Arrow(start=DOWN, end=UP, color=GREEN).scale(0.5).next_to(english[0], UP)))
		self.play(Write(h_text[0].scale(0.5).next_to(h_labels[1], RIGHT)))

		self.wait(1)
		self.play(Create(Arrow(start=LEFT, end=RIGHT).scale(0.5).next_to(h_text[0], RIGHT)))
		self.play(Write(h_labels[2].move_to(2*DOWN)))
		self.play(Write(english[1].scale(0.7).next_to(h_labels[2], 4*DOWN)))
		self.play(Create(Arrow(start=DOWN, end=UP, color=GREEN).scale(0.5).next_to(english[1], UP)))
		self.play(Write(h_text[1].scale(0.5).next_to(h_labels[2], RIGHT)))

		self.wait(1)
		self.play(Create(Arrow(start=LEFT, end=RIGHT).scale(0.5).next_to(h_text[1], RIGHT)))
		self.play(Write(h_labels[3].move_to(2*DOWN + 3.5*RIGHT)))
		self.play(Write(english[2].scale(0.7).next_to(h_labels[3], 4*DOWN)))
		self.play(Create(Arrow(start=DOWN, end=UP, color=GREEN).scale(0.5).next_to(english[2], UP)))
		self.play(Write(h_text[2].scale(0.5).next_to(h_labels[3], RIGHT)))

		self.wait(2)

		# Decoder Creation
		self.play(Write(s_labels[0].move_to(2*UP + 5.75*LEFT)))
		self.play(Write(s_text[0].scale(0.5).next_to(s_labels[0], RIGHT)))

		self.wait(1)
		self.play(Create(Arrow(start=LEFT, end=RIGHT).scale(0.5).next_to(s_text[0], RIGHT)))
		self.play(Write(s_labels[1].move_to(2*UP + 2.25*LEFT)))

		# Calculating annotations
		self.wait(2)
		self.play(Write(anno_label[0].move_to(3*LEFT)))
		self.play(Write(softmax_label[0].next_to(anno_label[0], RIGHT)))
		self.play(Write(hs_labels[0].scale(0.8).next_to(softmax_label[0], RIGHT)))
		self.play(Write(softmax_label[1].next_to(hs_labels[0], RIGHT)))

		self.wait(2)
		self.play(FadeOut(softmax_label[1]))
		self.play(ReplacementTransform(hs_labels[0], hm_text.scale(0.8).next_to(softmax_label[0], RIGHT)))
		self.play(Write(s_math[0].scale(0.8).next_to(hm_text)))
		self.play(FadeIn(softmax_label[1].next_to(s_math[0])))

		self.wait(2)
		self.play(ReplacementTransform(Group(hm_text, s_math[0]), anno_text[0].scale(0.8).next_to(softmax_label[0])))
		self.play(Write(softmax_label[1].next_to(anno_text[0])))

		self.wait(2)
		self.play(ReplacementTransform(Group(softmax_label[0], anno_text[0]), s_anno_text[0].scale(0.8).next_to(anno_label[0], RIGHT)), FadeOut(softmax_label[1]))

		# Calculating Context Vector
		self.wait(2)
		self.play(Group(anno_label[0], s_anno_text[0]).animate.shift(RIGHT*7))

		self.wait(1)
		self.play(Write(con_label.move_to(2*LEFT)))

		self.wait(2)
		self.play(ReplacementTransform(con_label, con_text[0].move_to(2*LEFT)))

		self.wait(2)
		self.play(ReplacementTransform(con_text[0], c_label[0].move_to(2*LEFT)))
		self.play(Write(c_text[0].scale(0.8).next_to(c_label[0], RIGHT)))

		# Calculating Output
		self.wait(2)
		self.play(Create(Arrow(start=DOWN, end=UP).scale(0.5).next_to(s_labels[1], DOWN)))
		self.play(Write(s_text[1].scale(0.5).next_to(s_labels[1], RIGHT)))

		self.wait(1)
		self.play(Create(Arrow(start=DOWN, end=UP, color=RED).scale(0.5).next_to(s_labels[1], UP)))
		self.play(Write(spanish[0].scale(0.7).next_to(s_labels[1], 5*UP)))

		self.wait(2)
