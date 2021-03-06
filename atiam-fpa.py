# %%
"""

PART 2 - Symbolic alignments and simple text dictionnaries

In this part, we will use our knowledge on computer structures to solve a very
well-known problem of string alignement. Hence, this part is split between
  1 - Implement a string alignment
  2 - Try to apply this to a collection of classical music pieces names
  3 - Develop your own more adapted procedure to have a matching inside large set

The set of classical music pieces is provided in the atiam-fpa.pkl file, which
is already loaded at this point of the script and contain two structures
    - composers         = Array of all composers in the database
    - composers_tracks  = Hashtable of tracks for a given composer

Some examples of the content of these structures

composers[23] => 'Abela, Placido'
composers[1210]  => 'Beethoven, Ludwig van'

composers_tracks['Abela, Placido'] => ['Ave Maria(Meditation on Prelude No. 1 by J.S.Bach)']
composers_tracks['Beethoven, Ludwig van'] => ['"Ode to Joy"  (Arrang.)', '10 National Airs with Variations, Op.107 ', ...]

composers_tracks['Beethoven, Ludwig van'][0] => '"Ode to Joy"  (Arrang.)'

"""

# %% Question 1 - Reimplementing the simple NW alignment

'''

Q-2.1 Here perform your Needleman-Wunsch (NW) implementation.
    - You can find the definition of the basic NW here
    https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
    - In this first version, we will be implementing the _basic_ gap costs
    - Remember to rely on a user-defined matrix for symbols distance

'''


def my_needleman_simple(str1, str2, matrix='atiam-fpa_alpha.dist', gap_open=-5, gap_extend=-5):
    score = 0
    ################
    # YOUR CODE HERE
    ################
    return ('', '', score)


# Reference code for testing
aligned = needleman_simple("CEELECANTH", "PELICAN",
                           matrix='atiam-fpa_alpha.dist', gap=-2)
print('Results for basic gap costs (linear)')
print(aligned[0])
print(aligned[1])
print('Score : ' + str(aligned[2]))

# %% Question 2 - Applying this to a collection of musical scores

################
# YOUR CODE HERE
################

'''

Q-2.2 Apply the NW algorithm between all tracks of each composer
    * For each track of a composer, compare to all remaining tracks of the same composer
    * Establish a cut criterion (what is the relevant similarity level ?) to only print relevant matches
    * Propose a set of matching tracks and save it through Pickle

'''

################
# YOUR CODE HERE
################

'''

Q-2.3 Extend your previous code so that it can compare
    * A given track to all tracks of all composers (full database)
    * You should see that the time taken is intractable (computational explosion)
    * Propose a method to avoid such a huge amount of computation
    * Establish a cut criterion (what is relevant similarity)
    * Propose a set of matching tracks and save it through Pickle

'''

################
# YOUR CODE HERE
################

# %%
"""

PART 3 - Extending the alignment algorithm and musical matching

You might have seen from the previous results that
        - Purely string matching on classical music names is not the best approach
        - This mostly comes from the fact that the importance of symbols is not the same
        - For instance
            "Symphony for orchestra in D minor"
            "Symphony for orchestra in E minor"
          Looks extremely close but the key is the most important symbol

Another flaw in our approach is that the NW algorithm treats all gaps
equivalently. Hence, it can put lots of small gaps everywhere.
Regarding alignement, it would be better to have long coherent gaps rather
than small ones. This is handled by a mecanism known as _affine gap penalty_
which separates the costs of either _opening_ or _extending_ a gap. This
is known as the Gotoh algorithm, which can be found here :
    - http://helios.mi.parisdescartes.fr/~lomn/Cours/BI/Material2019/gap-penalty-gotoh.pdf

"""
'''

Q-3.1 Extending to a true musical name matching
    * Start by exploring the collection for well-known composers, what do you see ?
    * Propose a new name matching algorithm adapted to classical music piece names
        - Can be based on a rule-based system
        - Can be a pre-processing for symbol finding and then an adapted weight matrix
        - Can be a local-alignement procedure
        (These are only given as indicative ideas ...)
    * Implement this new comparison procedure adapted to classical music piece names
    * Re-run your previous results (Q-2.2 and Q-2.3) with this procedure

'''

################
# YOUR CODE HERE
################

# Example of creating a dummy matrix
if DEV_MODE:
    dist = open('atiam-fpa_alpha.dist', 'w')
    dist.write('   ')
    for m1 in string.ascii_uppercase:
        dist.write(m1)
        if (m1 < 'Z'):
            dist.write('  ')
    dist.write('\n')
    for m1 in string.ascii_uppercase:
        dist.write(m1 + '  ')
        for m2 in string.ascii_uppercase:
            if (m2 == m1):
                dist.write('5  ')
            else:
                dist.write('-3  ')
        dist.write('\n')
    dist.close()


'''

Q-3.2 Extending the NW algorithm
    * Add the affine gap penalty to your original NW algorithm
    * You can use the Gotoh algorithm reference
    * Verify your code by using the provided compiled version

'''

################
# YOUR CODE HERE
################

aligned = needleman_affine(
    "CEELECANTH", "PELICAN", matrix='atiam-fpa_alpha.dist', gap_open=-5, gap_extend=-2)
print('Results for affine gap costs')
print(aligned[0])
print(aligned[1])
print('Score : ' + str(aligned[2]))


# %%
"""

PART 4 - Alignments between MIDI files and error-detection

Interestingly the problem of string alignment can be extended to the more global
problem of aligning any series of symbolic information (vectors). Therefore,
we can see that the natural extension of this problem is to align any sequence
of symbolic information.

This definition matches very neatly to the alignement of two musical scores
that can then be used as symbolic similarity between music, or score following.
However, this requires several key enhancements to the previous approach.
Furthermore, MIDI files gathered on the web are usually of poor quality and
require to be checked. Hence, here you will
    1 - Learn how to read and watch MIDI files
    2 - Explore their properties to perform some quality checking
    3 - Extend alignment to symbolic score alignement

To fasten the pace of your musical analysis, we will rely on the excellent
Music21 library, which provides all sorts of musicological analysis and
properties over symbolic scores. You will need to really perform this part
to go and read the documentation of this library online

"""

# %% Question 4 - Importing and plotting MIDI files (using Music21)


def get_start_time(el, measure_offset, quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(math.ceil(((measure_offset[el.measureNumber] or 0) + el.offset)*quantization))
    # Else, no time defined for this element and the functino return None


def get_end_time(el, measure_offset, quantization):
    if (el.offset is not None) and (el.measureNumber in measure_offset):
        return int(math.ceil(((measure_offset[el.measureNumber] or 0) + el.offset + el.duration.quarterLength)*quantization))
    # Else, no time defined for this element and the functino return None


def get_pianoroll_part(part, quantization):
    # Get the measure offsets
    measure_offset = {None: 0}
    for el in part.recurse(classFilter=('Measure')):
        measure_offset[el.measureNumber] = el.offset
    # Get the duration of the part
    duration_max = 0
    for el in part.recurse(classFilter=('Note', 'Rest')):
        t_end = get_end_time(el, measure_offset, quantization)
        if(t_end > duration_max):
            duration_max = t_end
    # Get the pitch and offset+duration
    piano_roll_part = np.zeros((128, math.ceil(duration_max)))
    for this_note in part.recurse(classFilter=('Note')):
        note_start = get_start_time(this_note, measure_offset, quantization)
        note_end = get_end_time(this_note, measure_offset, quantization)
        piano_roll_part[this_note.midi, note_start:note_end] = 1
    return piano_roll_part

# Here we provide a MIDI import function


def importMIDI(f):
    piece = converter.parse(f)
    all_parts = {}
    for part in piece.parts:
        print(part)
        try:
            track_name = part[0].bestName()
        except AttributeError:
            track_name = 'None'
        cur_part = get_pianoroll_part(part, 16)
        if (cur_part.shape[1] > 0):
            all_parts[track_name] = cur_part
    print('Returning')
    return piece, all_parts


################
# YOUR CODE HERE
################
'''

Q-4.1 Exploring MIDI properties

The Music21 library propose a lot of properties directly on the piece element,
but we also provide separately a dictionary containing for each part a matrix
representation (pianoroll) of the corresponding notes (without dynamics).
    - By relying on Music21 documentation (http://web.mit.edu/music21/doc/)
        * Explore various musicology properties proposed by the library
        * Check which could be used to assess the quality of MIDI files

'''

# Here a few properties that can be plotted ...
piece.plot('scatter', 'quarterLength', 'pitch')
piece.plot('scatterweighted', 'pitch', 'quarterLength')
piece.plot('histogram', 'pitchClass')
# Here is the list of all MIDI parts (with a pianoroll matrix)
for key, val in sorted(all_parts.items()):
    print('Instrument: %s has content: %s ' % (key, val))

################
# YOUR CODE HERE
################

'''

Q-4.2 Automatic evaluation of a MIDI file quality

One of the most pervasive problem with MIDI scores is that a large part of the
files that you can find on the internet are of rather low quality.
Based on your exploration in the previous questions and your own intuition,
    - Propose an automatic procedure that could evaluate the quality of a MIDI file.
    - Test how this could be used on a whole set of files

'''

################
# YOUR CODE HERE
################

'''

Q-4.3 Extending your alignment algorithm to MIDI scores

As explained earlier, our alignment algorithm can work with any set of symbols,
which of course include even complex scores. The whole trick here is to see
that the "distance matrix" previously used could simply be replaced by a
"distance function", which can represent the similarity between any elements
    - Propose a fit distance measures between two slices of pianorolls
    - Modify your previous algorithm so that it can use your distance
    - Modify the algorithm so that it can work with MIDI files
    - Apply your algorithm to sets of MIDI files

'''

################
# YOUR CODE HERE
################

# %% Just for preparing a random set of MIDIs to help you out
if DEV_MODE:
    nb_track = 0
    for val in np.random.randint(0, len(composers_tracks['Beethoven, Ludwig van']), 30):
        cur_track = composers_tracks['Beethoven, Ludwig van'][val]
        track_path = root + '/Beethoven, Ludwig van/' + \
            cur_track + '/' + cur_track + '.mid'
        os.system('cp ' + track_path + ' atiam-fpa/beethoven_' +
                  str(nb_track) + '.mid')
        print('cp "' + track_path + '" atiam-fpa/beethoven_' +
              str(nb_track) + '.mid')
        nb_track = nb_track + 1
