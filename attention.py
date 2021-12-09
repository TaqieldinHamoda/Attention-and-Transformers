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
			[1, 5, 1],
			[2, 0, -2],
		])

		s_vectors = np.array([
			[0, 1, 1],
			[1, 0, 1],
			[1, -1, -2],
		])

		anno1, s_anno1, con1 = self.context_vector(h_states, s_vectors[:, 0])
		anno2, s_anno2, con2 = self.context_vector(h_states, s_vectors[:, 1])

		# Displayed Text Portion
		h_colors = [GOLD, MAROON, BLUE]
		h_text =[
			 DecimalMatrix(
				[[h_states[0, 0]], [h_states[1, 0]], [h_states[2, 0]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]",
			),
			 DecimalMatrix(
				[[h_states[0, 1]], [h_states[1, 1]], [h_states[2, 1]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]",
			),
			 DecimalMatrix(
				[[h_states[0, 2]], [h_states[1, 2]], [h_states[2, 2]]],
				element_to_mobject_config={"num_decimal_places": 0},
				left_bracket="[",
				right_bracket="]",
			),
		]

		for i in range(len(h_colors)):
			h_text[i].set_color(h_colors[i])

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
			MathTex(r"h^{(1)} =", color=h_colors[0]),
			MathTex(r"h^{(2)} =", color=h_colors[1]),
			MathTex(r"h^{(3)} =", color=h_colors[2]),
		]

		s_labels = [
			MathTex(r"s^{(0)} ="),
			MathTex(r"s^{(1)} ="),
			MathTex(r"s^{(2)} ="),
		]

		h_matrix = Matrix(
			[[r"h^{(1)^{T}}"], [r"h^{(2)^{T}}"], [r"h^{(3)^{T}}"]],
            		element_alignment_corner=UL,
            		left_bracket="[",
		        right_bracket="]"
		)

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

		h_matrix.set_row_colors(*h_colors)
		hs_labels[0].set_row_colors(*h_colors)
		hs_labels[1].set_row_colors(*h_colors)

		anno_label = MathTex(r"\alpha =")
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

		anno_text[0].set_row_colors(*h_colors)
		anno_text[1].set_row_colors(*h_colors)

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

		s_anno_text[0].set_row_colors(*h_colors)
		s_anno_text[1].set_row_colors(*h_colors)

		# Get it, con_text vector
		c_label = MathTex(r"c =")

		con_text = [
			MathTex(
				r"[1]h^{(1)} + [2]h^{(2)} + [3]h^{(3)}".replace("[1]", f"{s_anno1[0]}").replace("[2]", f"{s_anno1[1]}").replace("[3]", f"{s_anno1[2]}"),
				substrings_to_isolate=("h^{(1)}", "h^{(2)}", "h^{(3)}"),
			),
			MathTex(
				r"[1]h^{(1)} + [2]h^{(2)} + [3]h^{(3)}".replace("[1]", f"{s_anno2[0]}").replace("[2]", f"{s_anno2[1]}").replace("[3]", f"{s_anno2[2]}"),
				substrings_to_isolate=("h^{(1)}", "h^{(2)}", "h^{(3)}"),
			),
		]

		for i in range(len(con_text)):
			con_text[i].set_color_by_tex("h^{(1)}", h_colors[0])
			con_text[i].set_color_by_tex("h^{(2)}", h_colors[1])
			con_text[i].set_color_by_tex("h^{(3)}", h_colors[2])

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

		softmax_label = [MathTex(r"softmax("), MathTex(r")")]

		english = [
			Text("\"I\"", color=GREEN),
			Text("\"Eat\"", color=GREEN),
			Text("\"Apples\"", color=GREEN),
		]

		spanish = [
			Text("\"Como\"", color=RED),
			Text("\"Manzanas\"", color=RED),
		]
		output_word = Text("\"Como\"", color=RED)


		# Creating Scene
		title = Text("Attention in RNNs", gradient=(BLUE, GREEN)).scale(1.5)
		self.play(Write(title))
		self.wait(2)
		self.play(FadeOut(title))

		# Encoder Creation
		self.wait(1)
		self.play(
			Write(english[0].scale(0.6).move_to(3.5*LEFT + 3.40*DOWN)),
			Write(english[1].scale(0.6).move_to(3.40*DOWN)),
			Write(english[2].scale(0.6).move_to(3.5*RIGHT + 3.40*DOWN))
		)

		self.play(Write(h_labels[0].move_to(2*DOWN + 5.75*LEFT)))

		self.play(Create(Arrow(start=LEFT, end=RIGHT).scale(0.5).next_to(h_labels[0], RIGHT)))
		self.play(Create(Arrow(start=DOWN, end=UP, color=GREEN).scale(0.5).next_to(english[0], UP)))
		self.play(Write(h_labels[1].move_to(2*DOWN + 3.5*LEFT)))
		self.play(Write(h_text[0].scale(0.5).next_to(h_labels[1], RIGHT)))

		self.play(Create(Arrow(start=LEFT, end=RIGHT).scale(0.5).next_to(h_text[0], RIGHT)))
		self.play(Create(Arrow(start=DOWN, end=UP, color=GREEN).scale(0.5).next_to(english[1], UP)))
		self.play(Write(h_labels[2].move_to(2*DOWN)))
		self.play(Write(h_text[1].scale(0.5).next_to(h_labels[2], RIGHT)))

		self.play(Create(Arrow(start=LEFT, end=RIGHT).scale(0.5).next_to(h_text[1], RIGHT)))
		self.play(Create(Arrow(start=DOWN, end=UP, color=GREEN).scale(0.5).next_to(english[2], UP)))
		self.play(Write(h_labels[3].move_to(2*DOWN + 3.5*RIGHT)))
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
		self.play(Write(anno_label.move_to(3*LEFT)))
		self.play(Write(softmax_label[0].next_to(anno_label, RIGHT)))
		self.play(Write(hs_labels[0].scale(0.8).next_to(softmax_label[0], RIGHT)))
		self.play(Write(softmax_label[1].next_to(hs_labels[0], RIGHT)))

		self.wait(2)
		self.play(
			ReplacementTransform(hs_labels[0], anno_text[0].scale(0.8).next_to(softmax_label[0])),
			softmax_label[1].animate.shift(LEFT)
		)

		self.wait(1)
		temp = MathTex(r"=")
		self.play(Write(temp.next_to(softmax_label[1], RIGHT)))
		self.play(Write(s_anno_text[0].scale(0.8).next_to(temp, RIGHT)))
		self.play(FadeOut(softmax_label[0]), FadeOut(anno_text[0]), FadeOut(temp), FadeOut(softmax_label[1]))
		self.play(anno_label.animate.shift(4.5*RIGHT))

		# Calculating Context Vector
		self.play(Group(anno_label, s_anno_text[0]).animate.shift(3*RIGHT))

		self.wait(1)
		self.play(Write(c_label.move_to(5.5*LEFT)))
		temp = MathTex(r"\alpha")
		self.play(Write(temp.next_to(c_label, RIGHT)))
		self.play(Write(h_matrix.next_to(temp, RIGHT)))

		self.play(FadeOut(anno_label), FadeOut(s_anno_text[0]))
		self.play(h_matrix.animate.shift(RIGHT))
		self.play(ReplacementTransform(temp, s_anno_text[0].next_to(c_label, RIGHT)))

		temp = MathTex(r"=")
		self.play(Write(temp.next_to(h_matrix, RIGHT)))
		self.play(Write(con_text[0].next_to(temp, RIGHT)))

		self.play(FadeOut(temp), FadeOut(s_anno_text[0]), FadeOut(h_matrix))
		self.play(con_text[0].animate.shift(4*LEFT))
		self.wait(1)

		# Arrow Shit
		arrow_1 = Arrow(start=(3.5*LEFT + 1.65*DOWN), end=(3.5*LEFT + 0.25*DOWN),
			color=h_colors[0],
			buff=0,
			max_tip_length_to_length_ratio=0.12,
			max_stroke_width_to_length_ratio=2,
		)
		arrow_2 = Arrow(start=(1.65*DOWN), end=(1.25*LEFT + 0.25*DOWN),
			color=h_colors[1],
			buff=0,
			max_tip_length_to_length_ratio=1,
			max_stroke_width_to_length_ratio=10,
		)
		arrow_3 = Arrow(start=(3.5*RIGHT + 1.65*DOWN), end=(RIGHT + 0.25*DOWN),
			color=h_colors[2],
			buff=0,
			max_tip_length_to_length_ratio=0,
			max_stroke_width_to_length_ratio=0.5,
		)

		self.play(Write(arrow_1))
		self.play(Write(arrow_2))
		self.play(Write(arrow_3))
		self.wait(2)

		self.play(FadeOut(con_text[0]), FadeOut(arrow_1), FadeOut(arrow_2), FadeOut(arrow_3))
		self.play(Write(c_text[0].scale(0.8).next_to(c_label, RIGHT)))
		self.play(Group(c_label, c_text[0]).animate.shift(3.5*RIGHT))

		# Calculating Output
		self.wait(2)
		temp_arrow = Arrow(start=DOWN, end=UP)
		self.play(Create(temp_arrow.scale(0.5).next_to(s_labels[1], DOWN)))
		self.play(Write(s_text[1].scale(0.5).next_to(s_labels[1], RIGHT)))

		self.wait(1)
		self.play(Create(Arrow(start=DOWN, end=UP, color=RED).scale(0.5).next_to(s_labels[1], UP)))
		self.play(Write(spanish[0].scale(0.6).next_to(s_labels[1], 5*UP)))

		self.wait(2)

		#################### Part 2 ######################
		# Clean Up Old Calculations
		self.play(FadeOut(Group(c_label, c_text[0], temp_arrow)))

		# Set Up Scene for S^{(2)}
		self.wait(1)
		temp_arrow = Arrow(start=LEFT, end=RIGHT)
		self.play(Create(temp_arrow.scale(0.5).next_to(s_text[1], RIGHT)))
		self.play(Write(s_labels[2].next_to(temp_arrow, RIGHT)))
		self.wait(0.5)

		# Repeat Calculations with new numbers
		# Calculating annotations
		self.wait(2)
		self.play(Write(anno_label.move_to(3*LEFT)))
		self.play(Write(softmax_label[0].next_to(anno_label, RIGHT)))
		self.play(Write(hs_labels[1].scale(0.8).next_to(softmax_label[0], RIGHT)))
		self.play(Write(softmax_label[1].next_to(hs_labels[1], RIGHT)))

		self.wait(2)
		self.play(
			ReplacementTransform(hs_labels[1], anno_text[1].scale(0.8).next_to(softmax_label[0])),
			softmax_label[1].animate.shift(LEFT)
		)

		self.wait(1)
		temp = MathTex(r"=")
		self.play(Write(temp.next_to(softmax_label[1], RIGHT)))
		self.play(Write(s_anno_text[1].scale(0.8).next_to(temp, RIGHT)))
		self.play(FadeOut(softmax_label[0]), FadeOut(anno_text[1]), FadeOut(temp), FadeOut(softmax_label[1]))
		self.play(anno_label.animate.shift(4.5*RIGHT))

		# Calculating Context Vector
		self.play(Group(anno_label, s_anno_text[1]).animate.shift(3*RIGHT))

		self.wait(1)
		self.play(Write(c_label.move_to(5.5*LEFT)))
		temp = MathTex(r"\alpha")
		self.play(Write(temp.next_to(c_label, RIGHT)))
		self.play(Write(h_matrix.next_to(temp, RIGHT)))

		self.play(FadeOut(anno_label), FadeOut(s_anno_text[1]))
		self.play(h_matrix.animate.shift(RIGHT))
		self.play(ReplacementTransform(temp, s_anno_text[1].next_to(c_label, RIGHT)))

		temp = MathTex(r"=")
		self.play(Write(temp.next_to(h_matrix, RIGHT)))
		self.play(Write(con_text[1].next_to(temp, RIGHT)))

		self.play(FadeOut(temp), FadeOut(s_anno_text[1]), FadeOut(h_matrix))
		self.play(con_text[1].animate.shift(4*LEFT))
		self.wait(1)

		# Arrow Shit
		arrow_1 = Arrow(start=(3.5*LEFT + 1.65*DOWN), end=(3.5*LEFT + 0.25*DOWN),
			color=h_colors[0],
			buff=0,
			max_tip_length_to_length_ratio=0,
			max_stroke_width_to_length_ratio=0.5,
		)
		arrow_2 = Arrow(start=(1.65*DOWN), end=(1.25*LEFT + 0.25*DOWN),
			color=h_colors[1],
			buff=0,
			max_tip_length_to_length_ratio=0.1,
			max_stroke_width_to_length_ratio=2,
		)
		arrow_3 = Arrow(start=(3.5*RIGHT + 1.65*DOWN), end=(RIGHT + 0.25*DOWN),
			color=h_colors[2],
			buff=0,
			max_tip_length_to_length_ratio=1,
			max_stroke_width_to_length_ratio=10,
		)

		self.play(Write(arrow_1))
		self.play(Write(arrow_2))
		self.play(Write(arrow_3))
		self.wait(2)

		self.play(FadeOut(con_text[1]), FadeOut(arrow_1), FadeOut(arrow_2), FadeOut(arrow_3))
		self.play(Write(c_text[1].scale(0.8).next_to(c_label, RIGHT)))
		self.play(Group(c_label, c_text[1]).animate.shift(3.4*RIGHT))

		# Calculating Output
		self.wait(2)
		temp_arrow = Arrow(start=DOWN, end=UP+RIGHT)
		self.play(Create(temp_arrow.scale(0.5).next_to(s_labels[2], DOWN + LEFT)))
		self.play(FadeIn(output_word.scale(0.6).next_to(s_labels[1], 5*UP)))
		self.play(output_word.animate.shift(3.25*DOWN + 3.25*RIGHT))
		self.play(Create(Arrow(start=DOWN, end=UP, color=RED).scale(0.5).next_to(s_labels[2], DOWN)))

		self.play(Write(s_text[2].scale(0.5).next_to(s_labels[2], RIGHT)))

		self.wait(1)
		self.play(Create(Arrow(start=DOWN, end=UP, color=RED).scale(0.5).next_to(s_labels[2], UP)))
		self.play(Write(spanish[1].scale(0.6).next_to(s_labels[2], 5*UP)))

		self.wait(2)
