project('gollum', 'c', default_options: ['default_library=both'])

math = meson.get_compiler('c').find_library('m', required : false)
gsl = dependency('gsl')
glib = dependency('glib-2.0')

gollum_lib = library('gollum', ['gollum.c', 'gollum.h'],
		     dependencies: glib,
		     install: true)

gollum_test = executable('test_gollum', 'test_gollum.c',
			 link_with: gollum_lib,
			 dependencies: glib)

test('Tests of gollum', gollum_test)
