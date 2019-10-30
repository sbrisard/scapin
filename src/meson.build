project('scapin', 'c', default_options: ['default_library=both'])

math = meson.get_compiler('c').find_library('m', required : false)
gsl = dependency('gsl')
glib = dependency('glib-2.0')

scapin_lib = library('scapin', ['scapin.c', 'scapin.h'],
		     dependencies: glib,
		     install: true)

scapin_test = executable('test_scapin', 'test_scapin.c',
			 link_with: scapin_lib,
			 dependencies: glib)

test('Tests of scapin', scapin_test)
