import freetype
import numpy as np
import subprocess
import itertools
from transmat import *
from multiprocessing.pool import ThreadPool
import os


def parse_point_tag(tag):
    bits = np.unpackbits(np.asarray([tag], np.uint8)).astype(np.bool)
    result = {
        'control': False,
        'cubic': False,
        'quad': False
    }
    if bits[-1]:
        result['control'] = False
    else:
        result['control'] = True
        # this seems to be not working
        # freetype is not giving the correct results
        if bits[-2]:
            result['cubic'] = True
        else:
            result['quad'] = True

    return result


# given a freetype glyph object, parses its outline
# for a specific typeface, the order of segments will always be the same
# it is the basis of the way we store additional points on curves
def parse_outline(glyph):
    """
    :type glyph: freetype.GlyphSlot
    """

    outline = glyph.outline

    # analyze points
    # depending on the length of the point segment, we can have
    # 2 - line segment
    # 3 - quadratic bezier curve
    # 4 - cubic bezier curve

    # first of all, extract all contours
    contours = []
    for i in range(len(outline.contours)):
        if i == 0:
            contours.append((0, outline.contours[0] + 1))
        else:
            contours.append((outline.contours[i - 1] + 1, outline.contours[i] + 1))

    # then, extract all contours
    # each of them is made up of sereral segments
    curves = []

    for (beg, end) in contours:
        # make sure it is a valid segment
        assert end - beg >= 2

        points = [outline.points[i] for i in range(beg, end)]
        tags = [parse_point_tag(outline.tags[i]) for i in range(beg, end)]

        i = j = 0
        if tags[0]['control']:
            raise RuntimeError(
                'This scenario is not implemented; see https://www.freetype.org/freetype2/docs/glyphs/glyphs-6.html')
        while i < len(points):
            assert not tags[i]['control']
            start_point = points[i]
            control_points = []
            end_point = None
            control_tag = None

            # find next on point
            for j in range(i + 1, len(points)):
                if not tags[j]['control']:
                    end_point = points[j]
                    break
                else:
                    control_points.append(points[j])
                    control_tag = tags[j]

            if end_point is None:
                if j == len(points) - 1:
                    end_point = points[0]
                    j += 1
                else:
                    raise RuntimeError('unable to parse contour: cannot find end point')

            i = j

            if len(control_points) > 2:
                assert control_tag['quad']
            if len(control_points) >= 2:
                if control_tag['quad']:
                    _start_point = np.asarray(start_point)
                    _control_point = _end_point = None
                    for k in range(len(control_points) - 1):
                        _control_point = control_points[k]
                        _end_point = (np.asarray(control_points[k]) + np.asarray(control_points[k + 1])) / 2.0
                        curves.append(np.asarray([_start_point, _control_point, _end_point], np.float))
                        _start_point = _end_point
                    curves.append(np.asarray([_start_point, control_points[-1], end_point], np.float))
                    continue

            curve = [start_point] + control_points + [end_point]
            curves.append(np.asarray(curve, np.float))

    return curves


def curve_to_tikz(curve, style_name='beziersty'):
    point_str = []
    num_points = curve.shape[0]
    for i in range(num_points):
        point_str.append('({},{})'.format(curve[i, 0], curve[i, 1]))
    if num_points == 2:
        return r'\draw[%s] %s--%s;' % (style_name, point_str[0], point_str[1])
    elif num_points == 3:
        return r'\draw[%s] %s .. controls %s .. %s;' % (style_name, point_str[0], point_str[1], point_str[2])
    elif num_points == 4:
        return r'\draw[%s] %s .. controls %s and %s .. %s;' % (
            style_name, point_str[0], point_str[1], point_str[2], point_str[3])
    else:
        raise RuntimeError('unsupported Bezier curve')

def transform_curve(curve, model, view, projection):
    # transform to homogeneous coord
    empty_coord = np.zeros((curve.shape[0], 2), np.float)
    homo_curve = np.concatenate([curve, empty_coord], axis=1)
    homo_curve[:, 3] = 1.0
    tf_curve = projection @ view @ model @ homo_curve.T
    tf_curve = tf_curve / tf_curve[-1, :]
    tf_curve = tf_curve.transpose()
    return tf_curve[:, :2]


if __name__ == "__main__":
    face = freetype.Face(r'C:\Windows\Fonts\cour.ttf')
    freetype.FT_CURVE_TAG(freetype.FT_CURVE_TAGS['FT_CURVE_TAG_ON'])
    face.set_char_size(30 * 64)

    offset_x_base = 1000
    offset_y_base = 1300
    curves1= []
    curves2 = []

    def load_glyph(g, target):
        face.load_char(g)
        target.append(parse_outline(face.glyph))

    for g in 'Alan 2021':
        load_glyph(g, curves1)

    for ind, curve in enumerate(curves1):
        now_offset = np.asarray([ind * offset_x_base, 0], np.float)
        for item in curve:
            item += now_offset

    for g in 'Xiang':
        load_glyph(g, curves2)

    offset_x_init = 600
    for ind, curve in enumerate(curves2):
        now_offset = np.asarray([offset_x_init + ind * offset_x_base, -offset_y_base], np.float)
        for item in curve:
            item += now_offset

    curves = curves1 + curves2
    canvas_size = (9, 5)

    tex_template = r'''
\documentclass[tikz]{standalone}
\usepackage{tikz}
\tikzset{
beziersty/.style={
    thick,
    line width=0.6pt,
    line cap=round
    }
}
\begin{document}
\begin{tikzpicture}
\node[minimum width=%f cm, 
    minimum height=%f cm,
    inner sep=0cm,
    outer sep=0cm] at (0cm,0cm) {};
%%s
\end{tikzpicture}
\end{document}
''' % canvas_size

    all_curves = list(itertools.chain(*curves))

    model = translate(-35.0, 0, 0.0) @ scale(*[0.010]*3)
    animation_ticks = np.linspace(0, np.pi * 2, 144, endpoint=True)
    saved_frames = []
    for ind, tick in enumerate(animation_ticks):
        x = np.cos(tick - np.pi / 2.0) * 20
        y = np.sin(tick - np.pi / 2.0) * 25
        z = 4 * np.cos(8.0 * tick) +  50.0
        eye = np.asarray([x, y, z])
        center = np.zeros(3, np.float)
        front = normalized(eye)
        right = normalized(np.cross(front, unit_y()))
        up = normalized(np.cross(right, front))
        view = look_at(eye, center, up)
        projection = perspective_projection(np.deg2rad(50.0), 1.0, 1.0, -1.0)
        transformed_curves = [transform_curve(curve, model, view, projection) for curve in all_curves]
        saved_frames.append((ind, transformed_curves))

    this_dir, _ = os.path.split(os.path.abspath(__file__))
    latex_dir = os.path.join(this_dir, 'frames')

    def compile_task(ind, transformed_curves):
        tikz_cmds = [curve_to_tikz(curve) for curve in transformed_curves]
        tex_content = tex_template % ('\n'.join(tikz_cmds),)
        os.chdir(latex_dir)
        filename = 'frame_{}.tex'.format(ind + 1)
        with open(filename, 'w') as outfile:
            outfile.write(tex_content)
        subprocess.run(['pdflatex', '-interaction=nonstopmode', filename], stdout=subprocess.DEVNULL)

    pool = ThreadPool(10)
    pool.starmap(compile_task, saved_frames)
